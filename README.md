# Visual Representation Learning for Robotic Manipulation

This project studies how to learn generalizable visual representation for robotics from videos of humans and natural language.

## Installation

First you can install a conda env from the r3m_base.yaml file [here](https://github.com/fairinternal/robolang_rep/blob/clean/r3m/r3m_base.yaml).

Then install from this directory with `pip install -e .`

You can test if it has installed correctly by running `import r3m` from a python shell.

## Using the representation

To use the model in your code simply run:
```
from r3m import load_r3m
r3m = load_r3m("resnet50") # resnet18, resnet34
r3m.eval()
```

Further example code to use a pre-trained representation is located in the example [here](https://github.com/fairinternal/robolang_rep/blob/clean/r3m/example.py)

## Training the representation

To train the representation run:

`python train_representation.py hydra/launcher=local hydra/output=local agent.langweight=1.0 agent.size=50 experiment=r3m_test dataset=ego4d doaug=rctraj agent.l1weight=0.00001 batch_size=16`
 
 Note, this reads data from my checkpoint directory. Still needs to be cleaned up more.
 
 
 
