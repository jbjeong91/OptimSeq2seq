import numpy as np
from typing import List, Tuple
import os
import const
import json

import math
import torch.nn as nn
import torch
import torch.nn.functional as F
# named SOTA + dec_attn + rel_attn
# torch_bool
try:
    torch_bool = torch.bool
except:
    torch_bool = torch.uint8

class Encoder(nn.Module):
    def __init__(self, config: const.Config, embedding: nn.modules.sparse.Embedding) -> None:
        super(Encoder, self).__init__()
        self.config = config
        #----------------------------------------------------------------------
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        #self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(d_model=100, nhead=10, dim_feedforward=200, dropout=0.3)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=2)
        #-------------------------------------------------------------------------

        self.hidden_size = config.encoder_num_units
        self.emb_size = config.embedding_dim
        self.words_number = config.words_number
        self.maxlen = config.max_sentence_length

        self.dropout = nn.Dropout(0.3)

        self.embedding = embedding
        self.cell_name = config.cell_name
        if config.cell_name == 'gru':
            self.rnn = nn.GRU(self.emb_size, self.hidden_size,  bidirectional=True, batch_first=True)
        elif config.cell_name == 'lstm':
            self.rnn = nn.LSTM(self.emb_size, self.hidden_size, bidirectional=True, batch_first=True)
        else:
            raise ValueError('cell name should be gru/lstm!')

    def forward(self, sentence: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:

        embedded = self.embedding(sentence)
        embedded = self.transformer_encoder(embedded)
        # embedded = self.dropout(embedded)

        if lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths=lengths, batch_first=True)

        output, hidden = self.rnn(embedded)
        #print('@@@@@@jeong_enc_result_hidden', np.array(hidden).size)

        if lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=self.maxlen, batch_first=True)

        output = (lambda a: sum(a)/2)(torch.split(output, self.hidden_size, dim=2))
        if self.cell_name == 'gru':
            hidden = (lambda a: sum(a)/2)(torch.split(hidden, 1, dim=0))
        elif self.cell_name == 'lstm':
            hidden = tuple(map(lambda state: sum(torch.split(state, 1, dim=0))/2, hidden))
        # hidden = (lambda a: sum(a)/2)(torch.split(hidden, 1, dim=0))

        return output, hidden


class Decoder(nn.Module):
    def __init__(self, config: const.Config, embedding: nn.modules.sparse.Embedding, device) -> None:
        super(Decoder, self).__init__()

        self.device = device
        self.config = config
        self.dropout = nn.Dropout(0.3)

        self.cell_name = config.cell_name
        self.decoder_type = config.decoder_type

        self.hidden_size = config.decoder_num_units
        self.emb_size = config.embedding_dim
        self.words_number = config.words_number
        self.maxlen = config.max_sentence_length
        self.decodelen = config.decoder_output_max_length

        #self.relation_eos = config.relation_number
        self.relation_number = config.relation_number

        self.word_embedding = embedding
        self.relation_embedding = nn.Embedding(config.relation_number + 1, config.embedding_dim)
        self.sos_embedding = nn.Embedding(1, config.embedding_dim)

        self.combine_inputs = nn.Linear(self.hidden_size + self.emb_size, self.emb_size)
        self.combine_inputs2 = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)#jeong

        self.combine_outputs = nn.Linear(self.emb_size + self.emb_size, self.emb_size)#jeong_1

        self.attn = nn.Linear(self.hidden_size * 2, 1)

        if self.cell_name == 'gru':
            self.rnn = nn.GRU(self.emb_size, self.hidden_size, batch_first=True)
        elif self.cell_name == 'lstm':
            self.rnn = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True, dropout=0.3)

        self.do_eos = nn.Linear(self.hidden_size, 1)
        self.do_predict = nn.Linear(self.hidden_size, self.relation_number)

        self.k_size = 3
        self.channel = 100
        self.combine_rel_inputs = nn.Linear(self.channel + self.emb_size, self.emb_size)
        self.do_rel_eos = nn.Linear(self.channel, 1)
        self.do_rel_predict = nn.Linear(self.channel, self.relation_number)

        self.fuse = nn.Linear(self.hidden_size * 2, 100)
        self.do_copy_linear = nn.Linear(100, 1)
        #----------------------------------------------------------------------------jeong_calc_new
        self.linear_ctx = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_query = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v = nn.Linear(self.hidden_size, 1)
        #----------------------------------------------------------------------------

    def calc_context(self, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:

        # decoder_state.size() == torch.Size([1, 100, 1000])
        # -> torch.Size([100, 1, 1000]) -> torch.Size([100, 80, 1000]) -cat-> torch.Size([100, 80, 2000])
        attn_weight = torch.cat((decoder_state.permute(1, 0, 2).expand_as(encoder_outputs), encoder_outputs), dim=2)
        attn_weight = F.softmax((self.attn(attn_weight)), dim=1)
        attn_applied = torch.bmm(attn_weight.permute(0, 2, 1), encoder_outputs).squeeze(1)

        return attn_applied

    def calc_context_new(self, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        #print('@@@@@@@@@@@@@@@@jeong_calc_dec',decoder_state.size()) #size([1,100,1000]
        #print('@@@@@@@@@@@@@@@@jeong_calc_enc', encoder_outputs.size())  #size([100,80,1000]
        decoder_state = decoder_state.permute(1,0,2) #size([100,1,1000]
        decoder_state = decoder_state.repeat(1, encoder_outputs.size()[1], 1)
        #print('@@@@@@@@@@@@@@@@jeong_calc_dec2', decoder_state.size()) #size([100,80,1000]
        uh = self.linear_ctx(encoder_outputs)
        wq = self.linear_query(decoder_state)
        wquh = torch.tanh(wq + uh)
        attn_weights = self.v(wquh).squeeze()
        #attn_weights.data.masked_fill_(src_mask.data, -float("inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # print(attn_weights)
        # print(src_mask)
        # print(torch.sum(attn_weights != attn_weights).any())
        # print('-----')

        # if torch.sum(attn_weights != attn_weights).any() > 0:
        #     exit()

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze()
        #print('@@@@@@@@@@@@@@@@jeong_calc_context',attn_applied.size())#size([100,1000])

        return attn_applied

    def extract_relation(self, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:

        # decoder_state.size() == torch.Size([1, 100, 1000])
        # -> torch.Size([100, 1, 1000]) -> torch.Size([100, 80, 1000])
        decoder_state = decoder_state.permute(1, 0, 2).expand_as(encoder_outputs)
        # torch.Size([100, 80, 1000]) -> torch.Size([100, 1000, 80])
        decoder_state = decoder_state.permute(0, 2, 1)
        attned_encoder_output = torch.bmm(encoder_outputs, decoder_state)  # [100, 80, 80]

        conv_emb = attned_encoder_output.unsqueeze(dim=1)  # [100,1,80,80]

        conv1 = torch.nn.Conv1d(in_channels=1, out_channels=self.channel, kernel_size=(self.k_size, self.maxlen),
                                stride=1).to(self.device)
        conv_emb = conv1(conv_emb)  # [100,10,76,1]
        conv_emb = conv_emb.squeeze(dim=-1)  # [100,10,76]
        conv_emb = F.relu(conv_emb)
        max_pooling = torch.nn.MaxPool2d((1, self.maxlen - (self.k_size -1)), stride=1)
        conv_emb = max_pooling(conv_emb) #[100,10,1]
        conv_emb = self.dropout(conv_emb)
        #print('@@@jeong_conv_emb', conv_emb.size())  # [100, 10, 1]
        conv_emb = conv_emb.squeeze(dim=-1)
        #print('@@@jeong_conv_emb',conv_emb.size()) #[100, 100]


        return conv_emb


    def do_copy(self, output: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:

        out = torch.cat((output.unsqueeze(1).expand_as(encoder_outputs), encoder_outputs), dim=2)
        out = F.selu(self.fuse(F.selu(out)))
        out = self.do_copy_linear(out).squeeze(2)
        # out = (self.do_copy_linear(out).squeeze(2))
        return out

    def _decode_step(self, rnn_cell: nn.modules,
                     emb: torch.Tensor,
                     decoder_state: torch.Tensor,
                     encoder_outputs: torch.Tensor,
                     first_entity_mask: torch.Tensor,
                     dec_states,
                     t) \
            -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        if self.cell_name == 'gru':
            decoder_state_h = decoder_state
        elif self.cell_name == 'lstm':
            decoder_state_h = decoder_state[0]
            #print('@@jeong_decoder_state_h', decoder_state_h.size())
            #print('@@jeong_decoder_state', decoder_state.size())
            #print('@@jeong_decoder_state[0]', decoder_state[0].size())
            #print('@@jeong_decoder_state[1]', decoder_state[1].size())
        else:
            raise ValueError('cell name should be lstm or gru')

        #print('@@jeong_decoder_state', decoder_state_h.size())
        #print('@@jeong_encoder_state', encoder_outputs.size())
        if t != 0:
            dec_states = dec_states
            dec_states.append(decoder_state_h.squeeze(0))
            dec_states = torch.stack(dec_states, dim=1)
            decoder_state_h = self.calc_context_new(decoder_state_h, dec_states)
            decoder_state_h = decoder_state_h.unsqueeze(0)
            #print('@@jeong_decoder_state_new', decoder_state_h.size())

        context = self.calc_context_new(decoder_state_h, encoder_outputs)
        if t%3==0:
            rel_context = self.extract_relation(decoder_state_h, encoder_outputs)
            output = self.combine_rel_inputs(torch.cat((emb, rel_context), dim=1))
        else:
            output = self.combine_inputs(torch.cat((emb, context), dim=1))
        output, decoder_state = rnn_cell(output.unsqueeze(1), decoder_state)
        output = output.squeeze()


        # eos_logits = F.selu(self.do_eos(output))
        # predict_logits = F.selu(self.do_predict(output))
        '''
        if t % 3 == 0:
            eos_logits = (self.do_eos(output))
            predict_logits = (self.do_predict(output))
            rel_logits, rel_eos_logits = self.extract_relation(decoder_state_h, encoder_outputs)
            eos_logits += rel_eos_logits
            predict_logits += rel_logits
            predict_logits = F.log_softmax(torch.cat((predict_logits, eos_logits), dim=1), dim=1)
        else:
            eos_logits = (self.do_eos(output))
            predict_logits = (self.do_predict(output))
            predict_logits = F.log_softmax(torch.cat((predict_logits, eos_logits), dim=1), dim=1)
        '''
        eos_logits = (self.do_eos(output))
        predict_logits = (self.do_predict(output))
        predict_logits = F.log_softmax(torch.cat((predict_logits, eos_logits), dim=1), dim=1)
        '''
        if t % 3 == 0:
            rel_logits, rel_eos_logits = self.extract_relation(decoder_state_h, encoder_outputs)
            #rel_predict_logits = F.log_softmax(torch.cat((rel_logits, rel_eos_logits), dim=1), dim=1)
            #predict_logits += rel_predict_logits
        '''
        copy_logits = self.do_copy(output, encoder_outputs)
        copy_logits = torch.cat((copy_logits, eos_logits), dim=1)
        copy_logits = F.log_softmax(copy_logits, dim=1)

        # # bug fix
        # copy_logits = torch.cat((copy_logits, eos_logits), dim=1)
        # first_entity_mask = torch.cat((first_entity_mask, torch.ones_like(eos_logits)), dim=1)
        #
        # copy_logits = F.softmax(copy_logits, dim=1)
        # copy_logits = copy_logits * first_entity_mask
        # copy_logits = torch.clamp(copy_logits, 1e-10, 1.)
        # copy_logits = torch.log(copy_logits)

        return (predict_logits, copy_logits), decoder_state

    def forward(self, *input):
        raise NotImplementedError('abstract method!')


class MultiDecoder(Decoder):

    def __init__(self, config: const.Config, embedding: nn.modules.sparse.Embedding, device) \
            -> None:
        super(MultiDecoder, self).__init__(config=config, embedding=embedding, device=device)
        self.decoder_cell_number = config.decoder_output_max_length // 3

        if self.cell_name == 'lstm':
            self.rnns = nn.ModuleList([nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
                                       for _ in range(self.decoder_cell_number)])
        elif self.cell_name == 'gru':
            self.rnns = nn.ModuleList([nn.GRU(self.emb_size, self.hidden_size, batch_first=True)
                                       for _ in range(self.decoder_cell_number)])
        else:
            raise NameError('lstm or gru!')

    def forward(self, sentence: torch.Tensor, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor) \
            -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        # sos = go = 0

        pred_action_list = []
        pred_logits_list = []

        go = torch.zeros(sentence.size()[0], dtype=torch.int64).to(self.device)
        output = self.sos_embedding(go)
        rel_output = None #jeong_1

        first_entity_mask = torch.ones(go.size()[0], self.maxlen).to(self.device)

        if self.cell_name == 'gru':
            previous_state = torch.zeros_like(decoder_state)
        elif self.cell_name == 'lstm':
            previous_state = tuple(map(torch.zeros_like, decoder_state))

        encoder_state = decoder_state

        for decoder in self.rnns:

            if self.cell_name == 'gru':
                decoder_state = (encoder_state + previous_state) / 2
            elif self.cell_name == 'lstm':
                decoder_state = ((encoder_state[0] + previous_state[0])/2, (encoder_state[1] + previous_state[1])/2)

            for t in range(3):

                bag, decoder_state = self._decode_step(decoder, output, decoder_state, encoder_outputs, first_entity_mask)
                predict_logits, copy_logits = bag

                if t % 3 == 0:
                    action_logits = predict_logits
                else:
                    action_logits = copy_logits

                max_action = torch.argmax(action_logits, dim=1).detach()

                pred_action_list.append(max_action)
                pred_logits_list.append(action_logits)

                # next time step
                if t % 3 == 0:
                    output = max_action
                    rel_output = self.relation_embedding(output)
                    output = self.relation_embedding(output)

                else:
                    copy_index = torch.zeros_like(sentence).scatter_(1, max_action.unsqueeze(1), 1).to(torch_bool)
                    output = sentence[copy_index]
                    #output = self.word_embedding(output)
                    # jeong_1 ->
                    output = torch.cat((rel_output, self.word_embedding(output)),dim=-1)
                    output = self.combine_outputs(output)

                if t % 3 == 1:
                    first_entity_mask = torch.ones(go.size()[0], self.maxlen + 1).to(self.device)

                    index = torch.zeros_like(first_entity_mask).scatter_(1, max_action.unsqueeze(1), 1).to(torch_bool)

                    first_entity_mask[index] = 0
                    first_entity_mask = first_entity_mask[:, :-1]

                else:
                    first_entity_mask = torch.ones(go.size()[0], self.maxlen).to(self.device)

            previous_state = decoder_state

        return pred_action_list, pred_logits_list


class OneDecoder(Decoder):

    def __init__(self, config: const.Config, embedding: nn.modules.sparse.Embedding, device) \
            -> None:
        super(OneDecoder, self).__init__(config=config, embedding=embedding, device=device)

    def forward(self, sentence: torch.Tensor, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor) \
            -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        # sos = go = 0

        pred_action_list = []
        pred_logits_list = []

        go = torch.zeros(sentence.size()[0], dtype=torch.int64).to(self.device)
        output = self.sos_embedding(go)
        rel_output = None  # jeong_1

        first_entity_mask = torch.ones(go.size()[0], self.maxlen).to(self.device)

        raw_outputs = []
        raw_outputs.append(decoder_state[0].squeeze(0))
        for t in range(self.decodelen):
            bag, decoder_state = self._decode_step(self.rnn, output, decoder_state, encoder_outputs, first_entity_mask, raw_outputs ,t)
            raw_outputs.append(decoder_state[0].squeeze(0))
            predict_logits, copy_logits = bag

            if t % 3 == 0:
                action_logits = predict_logits
            else:
                action_logits = copy_logits

            max_action = torch.argmax(action_logits, dim=1).detach()

            pred_action_list.append(max_action)
            pred_logits_list.append(action_logits)

            # next time step
            if t % 3 == 0:
                output = max_action
                rel_output = self.relation_embedding(output)
                output = self.relation_embedding(output)
                # --------------------------------------------------------
            elif t % 3 == 1:
                copy_index = torch.zeros_like(sentence).scatter_(1, max_action.unsqueeze(1), 1).to(torch_bool)
                output = sentence[copy_index]
                # jeong_1 ->
                # output = self.word_embedding(output)
                output = torch.cat((rel_output, self.word_embedding(output)), dim=-1)
                output = self.combine_outputs(output)
            else:
                copy_index = torch.zeros_like(sentence).scatter_(1, max_action.unsqueeze(1), 1).to(torch_bool)
                output = sentence[copy_index]
                output = self.word_embedding(output)
            # -------------------------------------------------------

            if t % 3 == 1:
                first_entity_mask = torch.ones(go.size()[0], self.maxlen + 1).to(self.device)

                index = torch.zeros_like(first_entity_mask).scatter_(1, max_action.unsqueeze(1), 1).to(torch_bool)

                first_entity_mask[index] = 0
                first_entity_mask = first_entity_mask[:, :-1]

            else:
                first_entity_mask = torch.ones(go.size()[0], self.maxlen).to(self.device)

        return pred_action_list, pred_logits_list



class Seq2seq(nn.Module):
    def __init__(self, config: const.Config, device, load_emb=False, update_emb=True):
        super(Seq2seq, self).__init__()

        self.device = device
        self.config = config

        self.emb_size = config.embedding_dim
        self.words_number = config.words_number
        self.maxlen = config.max_sentence_length

        self.word_embedding = nn.Embedding(self.words_number + 1, self.emb_size)
        if load_emb:
            self.load_pretrain_emb(config)
        self.word_embedding.weight.requires_grad = update_emb
        self.encoder = Encoder(config, embedding=self.word_embedding)

        if config.decoder_type == 'one':
            self.decoder = OneDecoder(config, embedding=self.word_embedding, device=device)
        elif config.decoder_type == 'multi':
            self.decoder = MultiDecoder(config, embedding=self.word_embedding, device=device)
        else:
            raise ValueError('decoder type one/multi!!')

        self.to(self.device)

    def load_pretrain_emb(self, config: const.Config) -> None:
        if os.path.isfile(config.words_id2vector_filename):
            # logger.info('Word Embedding init from %s' % config.words_id2vector_filename)
            print('load_embedding!')
            words_id2vec = json.load(open(config.words_id2vector_filename, 'r'))
            words_vectors = [0] * (len(words_id2vec) + 1)

            for i, key in enumerate(words_id2vec):
                words_vectors[int(key)] = words_id2vec[key]

            # words_vectors[len(words_id2vec) + 1] = [0] * len(words_id2vec[key])
            words_vectors[len(words_id2vec)] = [0] * len(words_id2vec[key])

            self.word_embedding.weight.data.copy_(torch.from_numpy(np.array(words_vectors)))

    def forward(self, sentence: torch.Tensor, sentence_eos: torch.Tensor, lengths: List[int]) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        o, h = self.encoder(sentence, lengths)
        pred_action_list, pred_logits_list = self.decoder(sentence=sentence_eos, decoder_state=h, encoder_outputs=o)

        return pred_action_list, pred_logits_list
