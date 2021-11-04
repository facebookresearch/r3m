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
import utils
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
    return hydra.utils.instantiate(cfg)

## Data Loader for Ego4D
class Ego4DBuffer(IterableDataset):
    def __init__(self, num_workers, source1, source2, alpha):
        self._num_workers = max(1, num_workers)
        self.alpha = alpha

        # Augmentations
        self.aug = torch.nn.Sequential(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.RandomAffine(20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            )

        # Load Data
        print("Ego4D")
        self.manifest = pd.read_csv("/private/home/surajn/data/ego4d/manifest.csv")
        print(self.manifest)
        self.mlen = len(self.manifest)

    def _sample(self):
        t0 = time.time()
        vid = np.random.randint(0, self.mlen)
        m = self.manifest.iloc[vid]
        vidlen = m["len"]
        txt = m["txt"]
        path = m["path"]

        start_ind = np.random.randint(1, 2 + int(self.alpha * vidlen))
        end_ind = np.random.randint(int((1-self.alpha) * vidlen)-1, vidlen)
        im0 = self.aug(torchvision.io.read_image(f"{path}/{start_ind:06}.jpg") / 255.0) * 255.0
        img = self.aug(torchvision.io.read_image(f"{path}/{end_ind:06}.jpg") / 255.0) * 255.0

        s1_ind = np.random.randint(2, vidlen)
        s0_ind = np.random.randint(1, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen+1)
        imts0 = self.aug(torchvision.io.read_image(f"{path}/{s0_ind:06}.jpg") / 255.0) * 255.0
        imts1 = self.aug(torchvision.io.read_image(f"{path}/{s1_ind:06}.jpg") / 255.0) * 255.0
        imts2 = self.aug(torchvision.io.read_image(f"{path}/{s2_ind:06}.jpg") / 255.0) * 255.0

        label = txt
        return (im0, img, imts0, imts1, imts2, label)

    def __iter__(self):
        while True:
            yield self._sample()

## Data Loader for Ground Truth Franka Kitchen Demo Data
class GTBuffer(IterableDataset):
    def __init__(self, num_workers, source1, source2, alpha, shuf, gt=False, alldata=None):
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.source = source1
        self.shuf = shuf
        self.gt = gt

        # Preprocess
        self.preprocess = torch.nn.Sequential(
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                )
        
        ## Augment when training
        if self.source == "train":
            self.aug = torch.nn.Sequential(
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    transforms.RandomAffine(20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                )
        else:
            self.aug = lambda a : a

        # Load Data
        self.cams = ["default", "left_cap2", "right_cap2"]
        self.tasks = ["kitchen_knob1_on-v3","kitchen_light_on-v3","kitchen_sdoor_open-v3",
                "kitchen_micro_open-v3","kitchen_ldoor_open-v3"]#,"kitchen_rdoor_open-v3"]
        self.instr = ["Turning the top left knob clockwise", "Switching the light on", 
                "Sliding the top right door open", "Opening the microwave", 
                "Opening the top left door"] #, "Opening the middle door"]
        if alldata is not None:
            self.all_demos, self.all_labels, self.all_states = alldata
        else:
            all_demos = []
            all_labels = []
            all_states = []
            for camera in ["default", "left_cap2", "right_cap2"]:
                for i, t in enumerate(self.tasks):
                    path = f"/private/home/surajn/code/vrl_private/vrl/hydra/expert_data/final_paths_multiview_rb_2k/{camera}/{t}.pickle"
                    demo_paths = pickle.load(open(path, 'rb'))
                    demo_paths = demo_paths[:1000]
                    print(len(demo_paths), i, t)
                    for p in demo_paths:
                        all_demos.append(p["images"])
                        all_states.append(p["observations"])
                        all_labels.append(self.instr[i])
            all_demos = np.stack(all_demos)
            all_states = np.stack(all_states)
            self.all_demos = all_demos
            self.all_labels = all_labels
            self.all_states = all_states
        

    def _sample(self):
        if self.source == "train":
            vid = np.random.randint(0, int(self.all_demos.shape[0]*0.8))
        elif self.source == "val":
            vid = np.random.randint(int(self.all_demos.shape[0]*0.8), self.all_demos.shape[0])
        t0 = time.time()
        vid = self.shuf[vid]
        ims = self.all_demos[vid]
        vidlen = ims.shape[0]
        

        start_ind = np.random.randint(0, 1 + int(self.alpha * vidlen))
        end_ind = np.random.randint(int((1-self.alpha) * vidlen)-1, vidlen)
        s1_ind = np.random.randint(1, vidlen-1)
        s0_ind = np.random.randint(0, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen)
        video = torch.FloatTensor(ims).permute(0, 3, 1, 2)

        im0 = self.aug(self.preprocess(video[start_ind]) / 255.0 ) * 255.0
        img = self.aug(self.preprocess(video[end_ind]) / 255.0 ) * 255.0
        imts0 = self.aug(self.preprocess(video[s0_ind]) / 255.0 ) * 255.0
        imts1 = self.aug(self.preprocess(video[s1_ind]) / 255.0 ) * 255.0
        imts2 = self.aug(self.preprocess(video[s2_ind]) / 255.0 ) * 255.0
        state = self.all_states[vid, s1_ind]
        label = self.all_labels[vid]
        if self.gt:
            return (im0, img, imts0, imts1, imts2, state)
        else:
            return (im0, img, imts0, imts1, imts2, label)

    def __iter__(self):
        while True:
            yield self._sample()


## Data Loader for SthSth Data
class VideoBuffer(IterableDataset):
    def __init__(self, num_workers, source1, source2, alpha):
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.source = source1
        
        # Preprocess
        self.preprocess = torch.nn.Sequential(
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                )

        ## Augment when training
        if self.source == "train":
            self.aug = torch.nn.Sequential(
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    transforms.RandomAffine(20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                )
        else:
            self.aug = lambda a : a

        # Load Data
        self.dt = pd.read_csv(f"/datasets01/SSV2/{source1}.csv", sep=" ")
        self.labels = {}
        f = open(f"/datasets01/SSV2/something-something-v2-{source2}.json")
        for label in json.load(f):
            self.labels[label["id"]] = label["label"]

    def _sample(self):
        vid = np.random.choice(self.dt["original_vido_id"])
        vidlen = self.dt.loc[self.dt["original_vido_id"] == vid]["frame_id"].max()
        start_ind = np.random.randint(1, 2 + int(self.alpha * vidlen))
        end_ind = np.random.randint(int((1-self.alpha) * vidlen)-1, vidlen)
        im0 = self.aug(self.preprocess(torchvision.io.read_image(f'/datasets01/SSV2/frames/{vid}/{vid}_{start_ind:06}.jpg')) / 255.0) * 255.0
        img = self.aug(self.preprocess(torchvision.io.read_image(f'/datasets01/SSV2/frames/{vid}/{vid}_{end_ind:06}.jpg')) / 255.0) * 255.0

        s1_ind = np.random.randint(2, vidlen)
        s0_ind = np.random.randint(1, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen+1)
        imts0 = self.aug(self.preprocess(torchvision.io.read_image(f'/datasets01/SSV2/frames/{vid}/{vid}_{s0_ind:06}.jpg')) / 255.0) * 255.0
        imts1 = self.aug(self.preprocess(torchvision.io.read_image(f'/datasets01/SSV2/frames/{vid}/{vid}_{s1_ind:06}.jpg')) / 255.0) * 255.0
        imts2 = self.aug(self.preprocess(torchvision.io.read_image(f'/datasets01/SSV2/frames/{vid}/{vid}_{s2_ind:06}.jpg')) / 255.0) * 255.0

        label = self.labels[str(vid)]
        return (im0, img, imts0, imts1, imts2, label)

    def __iter__(self):
        while True:
            yield self._sample()


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        print("Creating Dataloader")
        if self.cfg.dataset == "ego4d":
            train_iterable = Ego4DBuffer(self.cfg.replay_buffer_num_workers, "train", "train", alpha = self.cfg.alpha)
            ## Ego4D Val set is WIP
            val_iterable = train_iterable
        elif self.cfg.dataset == "gt":
            shuf = np.random.choice(200*5*3, 200*5*3, replace=False)
            train_iterable = GTBuffer(self.cfg.replay_buffer_num_workers, "train", "train",
                     alpha = self.cfg.alpha, shuf = shuf, gt = self.cfg.agent.gt)
            alldata = (train_iterable.all_demos, train_iterable.all_labels, train_iterable.all_states)
            val_iterable = GTBuffer(self.cfg.replay_buffer_num_workers, "val", "validation",
                     alpha=0, shuf = shuf, gt = self.cfg.agent.gt, alldata=alldata)
        else:
            train_iterable = VideoBuffer(self.cfg.replay_buffer_num_workers, "train", "train", alpha = self.cfg.alpha)
            val_iterable = VideoBuffer(self.cfg.replay_buffer_num_workers, "val", "validation", alpha=0)

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
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, cfg=self.cfg)

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
        self.model.eval_freq = eval_freq

        ## Training Loop
        print("Begin Training")
        while train_until_step(self.global_step):
            ## Sample Batch
            t0 = time.time()
            batch_frames_0, batch_frames_g, batch_frames_s0, batch_frames_s1, batch_frames_s2, batch_langs = next(self.train_loader)
            t1 = time.time()
            batch = (batch_frames_0.cuda(), batch_frames_g.cuda(), 
                    batch_frames_s0.cuda(), batch_frames_s1.cuda(), batch_frames_s2.cuda(), batch_langs)
            metrics = self.model.update(batch, self.global_step)
            t2 = time.time()
            self.logger.log_metrics(metrics, self.global_frame, ty='train')

            if self.global_step % 1 == 0:
                print(self.global_step, metrics)
                print(f'Sample time {t1-t0}, Update time {t2-t1}')
                
            if eval_every_step(self.global_step):
                with torch.no_grad():
                    batch_frames_0, batch_frames_g, batch_frames_s0, batch_frames_s1, batch_frames_s2, batch_langs = next(self.val_loader)
                    batch = (batch_frames_0.cuda(), batch_frames_g.cuda(), 
                            batch_frames_s0.cuda(), batch_frames_s1.cuda(), batch_frames_s2.cuda(), batch_langs)
                    metrics = self.model.update(batch, self.global_step, eval=True)
                    self.logger.log_metrics(metrics, self.global_frame, ty='eval')
                    print("EVAL", self.global_step, metrics)

                    self.save_snapshot()
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.global_step}.pt'
        global_snapshot =  self.work_dir / f'snapshot.pt'
        sdict = {}
        sdict["convnet"] = self.model.encoder.convnet.state_dict()
        if self.cfg.agent.lang_cond:
            sdict["encoder"] = self.model.encoder.state_dict()
        sdict["langpred"] = self.model.lang_pred.state_dict()
        torch.save(sdict, snapshot)
        sdict["global_step"] = self._global_step
        torch.save(sdict, global_snapshot)

    def load_snapshot(self, snapshot_path):
        payload = torch.load(snapshot_path)
        self.model.encoder.convnet.load_state_dict(payload['convnet'])
        if self.cfg.agent.lang_cond:
            self.model.encoder.load_state_dict(payload['encoder'])
        self.model.lang_pred.load_state_dict(payload['langpred'])
        try:
            self._global_step = payload['global_step']
        except:
            print("No global step found")

@hydra.main(config_path='cfgs', config_name='config_rep')
def main(cfg):
    from train_representation import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot(snapshot)
    workspace.train()


if __name__ == '__main__':
    main()