# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

import torchvision
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import pandas as pd
from robolang_rep import utils
from robolang_rep.trainer import Trainer
from robolang_rep.data_loaders import R3MBuffer, RoboNetBuffer, Ego4DBuffer, GTBuffer, VideoBuffer
from logger import Logger
import json
import time
import pickle
from torchvision.utils import save_image
import json
import random
import av

torch.backends.cudnn.benchmark = True


def make_network(cfg):
    model =  hydra.utils.instantiate(cfg)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)
    return model.cuda()

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        print("Creating Dataloader")
        if self.cfg.dataset == "robonet":
            sources = ["robonet"]
        elif self.cfg.dataset == "ego4d":
            sources = ["ego4d"]
        elif self.cfg.dataset == "sthsth":
            sources = ["sthsth"]
        elif self.cfg.dataset == "allhuman":
            sources = ["sthsth", "ego4d"]
        elif self.cfg.dataset == "all":
            sources = ["sthsth", "robonet", "ego4d"]

        train_iterable = R3MBuffer(self.cfg.replay_buffer_num_workers, "train", "train", 
                                    alpha = self.cfg.alpha, datasources=sources, doaug = self.cfg.doaug, simclr = self.cfg.simclr)
        val_iterable = R3MBuffer(self.cfg.replay_buffer_num_workers, "val", "validation", 
                                    alpha = 0, datasources=sources, doaug = 0, simclr = self.cfg.simclr)

        self.train_loader = iter(torch.utils.data.DataLoader(train_iterable,
                                         batch_size=self.cfg.batch_size,
                                         num_workers=self.cfg.replay_buffer_num_workers,
                                         pin_memory=True))
        self.val_loader = iter(torch.utils.data.DataLoader(val_iterable,
                                         batch_size=self.cfg.batch_size,
                                         num_workers=self.cfg.replay_buffer_num_workers,
                                         pin_memory=True))


        ## Init Model
        print("Initializing Model")
        self.model = make_network(cfg.agent)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        ## If reloading existing model
        if cfg.load_snap:
            print("LOADING", cfg.load_snap)
            self.load_snapshot(cfg.load_snap)

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=False, cfg=self.cfg)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.train_steps,
                                       1)
        eval_freq = self.cfg.eval_freq
        eval_every_step = utils.Every(eval_freq,
                                      1)
        trainer = Trainer(eval_freq)

        ## Training Loop
        print("Begin Training")
        while train_until_step(self.global_step):
            ## Sample Batch
            t0 = time.time()
            
            if self.cfg.simclr:
                bf1, bf2 = next(self.train_loader)
                t1 = time.time()
                batch = (bf1.cuda(), bf2.cuda())
                metrics, st = trainer.update_simclr(self.model, batch, self.global_step)
            else:
                batch_f, batch_langs = next(self.train_loader)
                t1 = time.time()
                metrics, st = trainer.update(self.model, (batch_f.cuda(), batch_langs), self.global_step)
            t2 = time.time()
            self.logger.log_metrics(metrics, self.global_frame, ty='train')

            if self.global_step % 10 == 0:
                print(self.global_step, metrics)
                print(f'Sample time {t1-t0}, Update time {t2-t1}')
                print(st)
                
            if eval_every_step(self.global_step):
                with torch.no_grad():
                    
                    if self.cfg.simclr:
                        bf1, bf2 = next(self.val_loader)
                        batch = (bf1.cuda(), bf2.cuda())
                        metrics, st = trainer.update_simclr(self.model, batch, self.global_step, eval=True)
                    else:
                        batch_f, batch_langs = next(self.val_loader)
                        metrics, st = trainer.update(self.model, (batch_f.cuda(), batch_langs), self.global_step, eval=True)
                    self.logger.log_metrics(metrics, self.global_frame, ty='eval')
                    print("EVAL", self.global_step, metrics)

                    self.save_snapshot()
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.global_step}.pt'
        global_snapshot =  self.work_dir / f'snapshot.pt'
        sdict = {}
        sdict["r3m"] = self.model.state_dict()
        torch.save(sdict, snapshot)
        sdict["global_step"] = self._global_step
        torch.save(sdict, global_snapshot)

    def load_snapshot(self, snapshot_path):
        payload = torch.load(snapshot_path)
        self.model.load_state_dict(payload['r3m'])
        try:
            self._global_step = payload['global_step']
        except:
            print("No global step found")

@hydra.main(config_path='cfgs', config_name='config_rep')
def main(cfg):
    from train_representation import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)

    restore_dir = ""
    # restore_dir = "/checkpoint/surajn/drqoutput/train_representation/2022-01-10_21-13-39"
    if restore_dir != "":
        last = str(root_dir.resolve()).split("/")[-1]
        snapshot = Path(f"{restore_dir}/{last}/snapshot.pt")
        print(snapshot)
        print("***")
        # assert(snapshot.exists())
    else:
        snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot(snapshot)
    workspace.train()


if __name__ == '__main__':
    main()