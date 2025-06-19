# BNN

BNN is the implementation of a binary-native and gradient-free learning algorithm for Binary Neural Networks.

BNN has been introduced in the [paper]() "".

*Disclaimer: we are working on the topic, so, expect changes in this repo.*

## External libraries used
You can install them using the provided `poetry` files.
First of all you need to install poetry (check [here](https://python-poetry.org/docs/#installation) for instructions) using:
```bash
pip install poetry
```
Then, you can install the dependencies with:
```bash
poetry install
```

## How to use
Clone the repo, install the dependencies and run the `src/train.py` script.

```bash
python3 src/train.py \
    --algo-layer ["our"/"baldassi"] \
    --dataset ["prototypes"/"fmnist"/"cifar10tl"/"imagenettetl"/"cifar100tl"] \
    --binarize-dataset/--no-binarize-dataset \
    --test-dim [int] \
    --layers [str] \
    --freeze-first/--no-freeze-first \
    --freeze-last/--no-freeze-last \
    --group-size [int] \
    --bs [int] \
    --epochs [int] \
    --prob-reinforcement [float] \
    --rob [float] \
    --seed [int] \
    --n-runs [int] \
    --device [int] \
    --log/--no-log \
```

### Comparisons with SGD and Adam
```bash
python3 {sgd/adam}_torch.py \
    --dataset ["prototypes"/"fmnist"/"bfmnist"/"cifar10tl"/"bcifar10tl"/"imagenettetl"/"bimagenettetl"/"cifar100tl"/"bcifar100tl"] \
    --layers [str] \
    --lr [float] \
    --seed [int] \
    --n-runs [int] \
    --device [int] \
```

### Results
To visualize the results as in the paper, look at the `VisualizeResults.ipynb` notebook.