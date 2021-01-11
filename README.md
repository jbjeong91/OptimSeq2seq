## Note

The codes is for my paper work, and it is modified based on the CopyMTL [repo](https://github.com/WindChimeRan/CopyMTL). Some parts from original repo have been deleted and modified.

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

- Train on GPU or CPU
`python main.py --gpu_use True --mode train --cell lstm`
`python main.py --gpu_use False --mode train --cell lstm`

- Test on GPU or CPU
`python main.py --gpu_use True --mode test --cell lstm`
`python main.py --gpu_use False --mode test --cell lstm`



