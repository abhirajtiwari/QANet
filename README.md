# QANet on SQuAD 2.0

- PyTorch implementation of the paper [QANET: Combining Local Convolution with Global Self-Attention for Reading Comprehension](https://arxiv.org/pdf/1804.09541.pdf) by Adams Wei Yu, David Dohan, Minh-Thang Luong, Rui Zhao, Kai Chen, Mohammad Norouzi, Quoc V. Le

## :bulb: Model Architecture
<img src="examples/qanet.png" width="742px" height="574px"/> <br>
Here are our [Notes](https://github.com/abhirajtiwari/SQuAD2/blob/master/examples/QANET_implementation.pdf) for the implementation

## :clipboard: Dependencies
```bash
git clone https://github.com/abhirajtiwari/SQuAD2.git
pip3 install -r requirements.txt
```
## Setup
1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
2. This also pre-processes the dataset for efficient data loading
```bash
python3 setup.py
```

## :wrench: Training
CLI args training `args.py`
```bash
python3 train.py -n train_run_1 --num_workers 4 --num_epochs 30 --eval_steps 50000 --batch_size 64 --hidden_size 128
```

To load the tensorboard
```bash
tensorboard --logdir save
```

## :books: Citing QANet
If you find QANet useful in your research, please consider citing:

```
@article{DBLP:journals/corr/abs-1804-09541,
  author    = {Adams Wei Yu and
               David Dohan and
               Minh{-}Thang Luong and
               Rui Zhao and
               Kai Chen and
               Mohammad Norouzi and
               Quoc V. Le},
  title     = {QANet: Combining Local Convolution with Global Self-Attention for
               Reading Comprehension},
  journal   = {CoRR},
  volume    = {abs/1804.09541},
  year      = {2018},
  url       = {http://arxiv.org/abs/1804.09541},
  archivePrefix = {arXiv},
  eprint    = {1804.09541},
  timestamp = {Mon, 13 Aug 2018 16:48:18 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1804-09541.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
If you liked our work, please leave a 🌟 <br>


<!-- Check on the following while training:
- --batch_size : default 64 (try 4)
- --num_workers : default 4 (try 2 or 1)
- --num_epochs : default 30 (try -1 to train forever)
- --eval_steps : default 50k (try anything lower, decide according to how much time it takes for the eval) 
- --hidden_size : default 128
- --load_path : default None
- --max_checkpoints : default 5

handle_json.py
- contains the split code
- open and set the splits sizes

<br>

To do on colab:
```bash
from google.colab import drive
drive.mount('/gdrive')

!git clone https://abhirajtiwari:password@github.com/abhirajtiwari/SQuAD2.git 
%cd SQuAD2
!git checkout dev_s

%%bash
apt install python3-pip
pip install ujson numpy spacy tensorboard tensorflow tensorboardX tqdm urllib3 torch torchvision

###############################################################

# open handle_json.py and change the splits
python3 handle_json.py # to make the splits, skip to use entire train.json


# Skip --train_url if training over entire train.json
python3 setup.py --train_url train-v2.0_1.json # to use the first split. To use the second split use 'train-v2.0_2.json'


python3 train.py -n baseline_train --num_workers 4 --num_epochs 5 --eval_steps 5000 --batch_size 16 --hidden_size 64 --load_path ./save/train/baseline_train-01/step_250024.pth.tar --max_checkpoints 10000 # set the args accordingly

###############################################################

!cp -r save/ /gdrive/My\ Drive/

%load_ext tensorboard
%tensorboard --logdir save --port 5678
```
-----------------------------
## Setup

1. Make sure you have [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) installed
    1. Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
    2. Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)

2. cd into src, run `conda env create -f environment.yml`
    1. This creates a Conda environment called `squad`

3. Run `conda activate squad`
    1. This activates the `squad` environment
    2. Do this each time you want to write/test your code
  
4. Run `python setup.py`
    1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
    2. This also pre-processes the dataset for efficient data loading
    3. For a MacBook Pro on the Stanford network, `setup.py` takes around 30 minutes total  

5. Browse the code in `train.py`
    1. The `train.py` script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
    2. You may find it helpful to browse the arguments provided by the starter code. Either look directly at the `parser.add_argument` lines in the source code, or run `python train.py -h`.

<br>
To train and open Tensorboard:

```bash
python3 trian.py -n baseline_train

tensorboard --logdir save --port 5678 # Start TensorBoard

python3 test.py -n test --load_path ./save/train/18-05-01/step_50000.pth.tar  #  For submission to leaderboard

```
-----------------------------
## Evaluating Model:
To run the evaluation:
```bash
python3 evaluate-v2.0.py <path_to_dev-v2.0> <path_to_predictions>
```

Sample input:
```bash
python3 evaluate-v2.0.py data/dev-v2.0.json dev-evaluate-v2.0-in1.txt
```

Sample output:
```
{
  "exact": 64.81091552261434,
  "f1": 67.60971132981278,
  "total": 11873,
  "HasAns_exact": 59.159919028340084,
  "HasAns_f1": 64.7655368790259,
  "HasAns_total": 5928,
  "NoAns_exact": 70.4457527333894,
  "NoAns_f1": 70.4457527333894,
  "NoAns_total": 5945
} 
```
-->
