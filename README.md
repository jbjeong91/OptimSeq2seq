# 수정중..

## Environment

python3

pytorch 0.4.0 -- 1.3.1

## Modify the Data path

This repo initially contain webnlg, you can run the code directly.
NYT dataset need to be downloaded and to be placed in proper path. see **const.py**.

The pre-processed data is avaliable in:

WebNLG dataset:
 https://drive.google.com/open?id=1zISxYa-8ROe2Zv8iRc82jY9QsQrfY1Vj

NYT dataset:
 https://drive.google.com/open?id=10f24s9gM7NdyO3z5OqQxJgYud4NnCJg3
 


## Run

`python main.py --gpu 0 --mode train --cell lstm --decoder_type one`

`python main.py --gpu 0 --mode test --cell lstm --decoder_type one`



