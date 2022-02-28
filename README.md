# Visual Representation Learning for Robotic Manipulation

This project studies how to learn generalizable visual representation for robotics from videos of humans and natural language.

## Installation

First you can install a conda env from the environment.yml file [here](https://github.com/fairinternal/robolang_rep/blob/clean/robolang_rep/environment.yml).

Then install from this directory with `pip install -e .`

You can test if it has installed correctly by running `import robolang_rep` from a python shell.

## Using the representation

Example code to use a pre-trained representation is located in the example [here](https://github.com/fairinternal/robolang_rep/blob/clean/robolang_rep/example.py)

The model to use is located at `/checkpoint/surajn/drqoutput/train_representation/2022-01-24_11-10-28/10_agent.finetunelang=0,agent.l1weight=1e-05,agent.langtype=lorel,agent.langweight=1.0,agent.size=50,batch_size=16,dataset=ego4d,doaug=rctraj,experiment=rep_0124/snapshot_1000000.pt`

## Training the representation

To train the representation run:

`python train_representation.py hydra/launcher=local hydra/output=local agent.langweight=1.0 agent.size=50 experiment=r3m_test dataset=ego4d doaug=rctraj agent.l1weight=0.00001 batch_size=16`
 
 Note, this reads data from my checkpoint directory. Still needs to be cleaned up more.
 
 
 
