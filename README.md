# SQuAD2.0

## Evaluating Model:
To run the evaluation:
```bash
python3 evaluate-v2.0.py <path_to_dev-v2.0> <path_to_predictions>
```

Sample input:
```bash
python3 evaluate-v2.0.py dev-v2.0.json dev-evaluate-v2.0-in1.txt
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
```
