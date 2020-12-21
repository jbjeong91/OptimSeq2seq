import numpy as np
from typing import List, Tuple
import os
import const
import json

import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable


def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations, device):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps))).to(device=device)

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps)).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        return v


class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module, device):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim)).to(device=device)
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)
        #print('@@@@@jeong_caps_output',caps_output.size())#[10,1007616,1,8]
        #print('@@@@jeong_weight', self.weights.size())#[1152,8,3952]
        u_predict = caps_output.matmul(self.weights)

        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        v = self.routing_module(u_predict)
        return v


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride, device):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride).to(device)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        #print('@@@@@@jeong_in', input.size())
        out = self.conv(input)
        #print('@@@@@@jeong_out',out.size())
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)
        #print('@@@@@@jeong_out2', out.size())

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        #print('@@@@@@jeong_out3', out.size())
        out = squash(out)
        return out


class CapsNet(nn.Module):
    def __init__(self, config: const.Config, routing_iterations, device):
        super(CapsNet, self).__init__()
        self.final_dim = 16
        self.n_classes = config.triple_number
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(int(config.max_sentence_length/4),int(config.decoder_num_units/4)), stride=(1,4)).to(self.device)
        self.primaryCaps = PrimaryCapsLayer(input_channels=256, output_caps=32, output_dim=8, kernel_size=10, stride=(10,10), device=device)  # outputs 6*6
        self.num_primaryCaps = 32*9*12#32 * 6 * 6
        routing_module = AgreementRouting(self.num_primaryCaps, self.n_classes, routing_iterations, device)
        self.digitCaps = CapsLayer(self.num_primaryCaps, 8, self.n_classes, self.final_dim, routing_module, device)
        self.do_predict = nn.Linear(self.final_dim, config.relation_number).to(self.device)
        self.do_eos = nn.Linear(self.final_dim, 1).to(self.device)
        '''
        conv_emb = encoder_outputs
        conv_emb = torch.unsqueeze(conv_emb, 1)
        #print('@@@@@@@@@jeong_conv_emb', conv_emb.size())
        conv = torch.nn.Conv2d(in_channels=1, out_channels=self.relation_number, kernel_size=(self.config.max_sentence_length, 1), stride=1, padding=0).to(self.device)
        conv_emb = conv(conv_emb)
        conv_emb = torch.squeeze(conv_emb, 2)
        #print('@@@@@@@@@jeong_conv_emb', conv_emb.size())
        conv_emb = self.transformer_encoder(conv_emb)
        #max_pooling = torch.nn.MaxPool1d(3, stride=1, padding=1)
        #conv_emb = max_pooling(conv_emb)
        # print('@@@@@@@@@jeong_basic_emb',basic_emd.size())#[100, 80, 1000])
        '''

    def forward(self, input):
        #input = [batch = 10, 1, 80, 1000]
        x = self.conv1(input)
        #print('###jeong_x1',x.size()) #[10,512, 72, 992]
        x = F.relu(x)
        #print('###jeong_x2', x.size()) #[10,512, 72, 992]
        x = self.primaryCaps(x)
        #print('###jeong_x3', x.size()) #[10,1007616, 8]
        x = self.digitCaps(x)
        #print('###jeong_x4', x.size())
        #probs = x.pow(2).sum(dim=2).sqrt()
        #print('###jeong_probs',probs.size())
        #jeong->
        predict_logits = self.do_predict(x)
        eos_logits = self.do_eos(x)
        x = F.log_softmax(torch.cat((predict_logits, eos_logits), dim=-1), dim=-1)
        return x#, probs

class ReconstructionNet(nn.Module):
    def __init__(self, n_dim=16, n_classes=10):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x, target):
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

class CapsNetWithReconstruction(nn.Module):
    def __init__(self, capsnet, reconstruction_net):
        super(CapsNetWithReconstruction, self).__init__()
        self.capsnet = capsnet
        self.reconstruction_net = reconstruction_net

    def forward(self, x, target):
        x, probs = self.capsnet(x)
        reconstruction = self.reconstruction_net(x, target)
        return reconstruction, probs

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
        encoder_layers = TransformerEncoderLayer(d_model=100, nhead=50, dim_feedforward=200, dropout=0.3)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=2)
        #-------------------------------------------------------------------------

        self.hidden_size = config.encoder_num_units
        self.emb_size = config.embedding_dim
        self.words_number = config.words_number
        self.maxlen = config.max_sentence_length

        self.dropout = nn.Dropout(0.1)

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
        #----------------------------------------------------------------------
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        #self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(d_model=1000, nhead=50, dim_feedforward=1000, dropout=0.3)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=2)
        #-------------------------------------------------------------------------


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
        self.combine_inputs2 = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)  # jeong
        self.combine_outputs = nn.Linear(self.emb_size + self.emb_size, self.emb_size)#jeong_1
        self.W_cap_prob = nn.Linear(self.relation_number, self.relation_number)  # jeong_1

        self.attn = nn.Linear(self.hidden_size * 2, 1)
        #----------------------------------------------------------------------------jeong_calc_new
        self.linear_ctx = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_query = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v = nn.Linear(self.hidden_size, 1)
        #----------------------------------------------------------------------------
        if self.cell_name == 'gru':
            self.rnn = nn.GRU(self.emb_size, self.hidden_size, batch_first=True)
        elif self.cell_name == 'lstm':
            self.rnn = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)

        self.do_eos = nn.Linear(self.hidden_size, 1)
        self.do_predict = nn.Linear(self.hidden_size, self.relation_number)

        self.fuse = nn.Linear(self.hidden_size * 2, 100)
        self.do_copy_linear = nn.Linear(100, 1)

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
        #decoder_state = decoder_state.repeat(1, self.maxlen, 1)
        decoder_state = decoder_state.repeat(1, self.relation_number, 1)
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

    def do_copy(self, output: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:

        out = torch.cat((output.unsqueeze(1).expand_as(encoder_outputs), encoder_outputs), dim=2)
        out = F.selu(self.fuse(F.selu(out)))
        out = self.do_copy_linear(out).squeeze(2)
        # out = (self.do_copy_linear(out).squeeze(2))
        return out

    def _decode_step_new(self, rnn_cell: nn.modules,
                     emb: torch.Tensor,
                     decoder_state: torch.Tensor,
                     encoder_outputs: torch.Tensor,
                     first_entity_mask: torch.Tensor,
                     conv_emb: torch.Tensor,
                     time_step: torch.int) \
            -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        if self.cell_name == 'gru':
            decoder_state_h = decoder_state
        elif self.cell_name == 'lstm':
            decoder_state_h = decoder_state[0]
        else:
            raise ValueError('cell name should be lstm or gru')

        #print('@@@@@@@@@@jeong_decoder_state_h', decoder_state_h.size()) #[1, batch=80, dom=1000]
        #print('@@@@@@@@@@jeong_encoder_outputs', encoder_outputs.size())#[batch, seq_len=80, dim=1000]
        context = self.calc_context(decoder_state_h, encoder_outputs)
        #print('@@@@@@@@@@jeong_context', context.size())#[seq_len=80, dim=1000]
        rel_attn = self.calc_context(decoder_state_h, conv_emb)
        #rel_attn = self.combine_inputs2(torch.cat((rel_attn, context), dim=-1))

        if time_step % 3 == 0:
            output = self.combine_inputs(torch.cat((emb, rel_attn), dim=1))
            output, decoder_state = rnn_cell(output.unsqueeze(1), decoder_state)
            #output = output.squeeze().unsqueeze(dim=0)
            #print('@@@@@@@@@@jeong_output1',output.size())#[1,50,1000]
            #print('@@@@@@@@@@jeong_conv_emv', conv_emb.size())  # [50,247,1000]
            #output = self.calc_context_new(output, conv_emb)
            output = output.squeeze()
            #print('@@@@@@@@@@jeong_output2', output.size())
        else:
            output = self.combine_inputs(torch.cat((emb, context), dim=1))
            output, decoder_state = rnn_cell(output.unsqueeze(1), decoder_state)
            output = output.squeeze()

        # eos_logits = F.selu(self.do_eos(output))
        # predict_logits = F.selu(self.do_predict(output))
        eos_logits = (self.do_eos(output))
        predict_logits = (self.do_predict(output))


        predict_logits = F.log_softmax(torch.cat((predict_logits, eos_logits), dim=1), dim=1)

        copy_logits = self.do_copy(output, encoder_outputs)

        # assert copy_logits.size() == first_entity_mask.size()
        # original
        # copy_logits = copy_logits * first_entity_mask
        # copy_logits = copy_logits

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

    def _decode_step_rel_attn(self, rnn_cell: nn.modules,
                     emb: torch.Tensor,
                     decoder_state: torch.Tensor,
                     encoder_outputs: torch.Tensor,
                     first_entity_mask: torch.Tensor,
                              cap_logit: torch.Tensor) \
            -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        if self.cell_name == 'gru':
            decoder_state_h = decoder_state
        elif self.cell_name == 'lstm':
            decoder_state_h = decoder_state[0]
        else:
            raise ValueError('cell name should be lstm or gru')

        context = self.calc_context(decoder_state_h, encoder_outputs)

        output = self.combine_inputs(torch.cat((emb, context), dim=1))

        output, decoder_state = rnn_cell(output.unsqueeze(1), decoder_state)

        output = output.squeeze()

        # eos_logits = F.selu(self.do_eos(output))
        # predict_logits = F.selu(self.do_predict(output))
        eos_logits = (self.do_eos(output))

        #------------------------------------------------
        # predict_logits = (self.do_predict(output))
        predict_logits = (self.do_predict(output)) + self.W_cap_prob(cap_logit)
        #------------------------------------------------
        #print('######jeong_predict_logits', predict_logits.size())

        predict_logits = F.log_softmax(torch.cat((predict_logits, eos_logits), dim=1), dim=1)

        copy_logits = self.do_copy(output, encoder_outputs)

        # assert copy_logits.size() == first_entity_mask.size()
        # original
        # copy_logits = copy_logits * first_entity_mask
        # copy_logits = copy_logits

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

    def _decode_step(self, rnn_cell: nn.modules,
                     emb: torch.Tensor,
                     decoder_state: torch.Tensor,
                     encoder_outputs: torch.Tensor,
                     first_entity_mask: torch.Tensor) \
            -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        if self.cell_name == 'gru':
            decoder_state_h = decoder_state
        elif self.cell_name == 'lstm':
            decoder_state_h = decoder_state[0]
        else:
            raise ValueError('cell name should be lstm or gru')

        context = self.calc_context(decoder_state_h, encoder_outputs)

        output = self.combine_inputs(torch.cat((emb, context), dim=1))

        output, decoder_state = rnn_cell(output.unsqueeze(1), decoder_state)

        output = output.squeeze()

        # eos_logits = F.selu(self.do_eos(output))
        # predict_logits = F.selu(self.do_predict(output))
        eos_logits = (self.do_eos(output))
        predict_logits = (self.do_predict(output))
        #print('######jeong_predict_logits', predict_logits.size())

        predict_logits = F.log_softmax(torch.cat((predict_logits, eos_logits), dim=1), dim=1)

        copy_logits = self.do_copy(output, encoder_outputs)

        # assert copy_logits.size() == first_entity_mask.size()
        # original
        # copy_logits = copy_logits * first_entity_mask
        # copy_logits = copy_logits

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
        rel_output = self.sos_embedding(go) #jeong_1

        first_entity_mask = torch.ones(go.size()[0], self.maxlen).to(self.device)

        if self.cell_name == 'gru':
            previous_state = torch.zeros_like(decoder_state)
        elif self.cell_name == 'lstm':
            previous_state = tuple(map(torch.zeros_like, decoder_state))

        encoder_state = decoder_state

        reconstruction_model = ReconstructionNet(16, 10)
        capsnet = CapsNet(config=self.config, routing_iterations=3, device=self.device)
        #capsnet = CapsNetWithReconstruction(capsnet, reconstruction_model)
        encoder_outputs_redim = torch.unsqueeze(encoder_outputs, 1)
        caps_result = capsnet(encoder_outputs_redim)
        #cabs_logits = caps_result[0]
        #cabs_prob = caps_result[1]
        #print('@@@@@@@@@jeong_cabs_emb',caps_result)
        caps_result = torch.unbind(caps_result, dim=1)

        count = -1
        for decoder in self.rnns:
            count +=1
            #----------------------------------------------------------------------------------------------------------------
            if self.cell_name == 'gru':
                decoder_state = (encoder_state + previous_state) / 2
                #decoder_state = encoder_state
            elif self.cell_name == 'lstm':
                decoder_state = ((encoder_state[0] + previous_state[0])/2, (encoder_state[1] + previous_state[1])/2)
                #decoder_state = (encoder_state[0], encoder_state[1])

            for t in range(2):
                if t == 0:
                    pred_logits_list.append(caps_result[count])
                    max_caps_result = torch.argmax(caps_result[count], dim=1)
                    pred_action_list.append(max_caps_result)
                    #output = rel_output#previous_rel
                    output = self.relation_embedding(max_caps_result)
                bag, decoder_state = self._decode_step(decoder, output, decoder_state, encoder_outputs, first_entity_mask)
                predict_logits, copy_logits = bag

                '''
                if t % 2 == 0:
                    #action_logits = predict_logits
                    #jeong->
                    action_logits = predict_logits + caps_result[count]
                else:
                    action_logits = copy_logits
                '''
                action_logits = copy_logits
                max_action = torch.argmax(action_logits, dim=1).detach()

                pred_action_list.append(max_action)
                pred_logits_list.append(action_logits)

                # next time step
                #if t % 2 == 0:
                #    output = max_action
                #    rel_output = self.relation_embedding(output)
                #    output = self.relation_embedding(output)

                #else:
                copy_index = torch.zeros_like(sentence).scatter_(1, max_action.unsqueeze(1), 1).to(torch_bool)
                output = sentence[copy_index]
                output = self.word_embedding(output)
                # jeong_1 ->
                #output = torch.cat((rel_output, self.word_embedding(output)),dim=-1)
                #output = self.combine_outputs(output)

                #if t % 2 == 1:
                if t % 2 == 0:
                    first_entity_mask = torch.ones(go.size()[0], self.maxlen + 1).to(self.device)

                    index = torch.zeros_like(first_entity_mask).scatter_(1, max_action.unsqueeze(1), 1).to(torch_bool)

                    first_entity_mask[index] = 0
                    first_entity_mask = first_entity_mask[:, :-1]

                else:
                    first_entity_mask = torch.ones(go.size()[0], self.maxlen).to(self.device)

            previous_state = decoder_state
        #---------------------------------------------------------------------------------------------------------------
        return pred_action_list, pred_logits_list


class OneDecoder(Decoder):

    def __init__(self, config: const.Config, embedding: nn.modules.sparse.Embedding, device) \
            -> None:
        super(OneDecoder, self).__init__(config=config, embedding=embedding, device=device)
        self.config = config
        self.device = device
    def forward(self, sentence: torch.Tensor, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor) \
            -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        # sos = go = 0

        pred_action_list = []
        pred_logits_list = []

        go = torch.zeros(sentence.size()[0], dtype=torch.int64).to(self.device)
        output = self.sos_embedding(go)
        rel_output = None  # jeong_1

        first_entity_mask = torch.ones(go.size()[0], self.maxlen).to(self.device)

        #---------------------------------------------------------------------------------------jeong_5
        reconstruction_model = ReconstructionNet(16, 10)
        capsnet = CapsNet(config=self.config, routing_iterations=3, device=self.device)
        #capsnet = CapsNetWithReconstruction(capsnet, reconstruction_model)
        encoder_outputs_redim = torch.unsqueeze(encoder_outputs, 1)
        caps_result = capsnet(encoder_outputs_redim)
        #cabs_logits = caps_result[0]
        #cabs_prob = caps_result[1]
        #print('@@@@@@@@@jeong_cabs_emb',caps_result)
        caps_result = torch.unbind(caps_result, dim=1)
        #--------------------------------------------------------------------------------------------------------
        count = -1
        for t in range(self.decodelen):
            if t%3==0:
                count +=1
                max_caps_result = torch.argmax(caps_result[count], dim=1)

                output = torch.cat((output, self.relation_embedding(max_caps_result)), dim=-1)
                output = self.combine_outputs(output)

            bag, decoder_state = self._decode_step(self.rnn, output, decoder_state, encoder_outputs, first_entity_mask)
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
                # output = self.word_embedding(output)
                # jeong_1 ->
                output = torch.cat((rel_output, self.word_embedding(output)), dim=-1)
                output = self.combine_outputs(output)

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
