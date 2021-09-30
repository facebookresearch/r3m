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
from torchvision.utils import save_image


torch.backends.cudnn.benchmark = True


def make_network(cfg):
    return hydra.utils.instantiate(cfg)

class VideoBuffer(IterableDataset):
    def __init__(self, num_workers, source1, source2, alpha):
        self._num_workers = max(1, num_workers)
        self.alpha = alpha

        # Preprocess
        self.preprocess = torch.nn.Sequential(
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                )

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
        im0 = self.preprocess(torchvision.io.read_image(f'/datasets01/SSV2/frames/{vid}/{vid}_{start_ind:06}.jpg'))
        img = self.preprocess(torchvision.io.read_image(f'/datasets01/SSV2/frames/{vid}/{vid}_{end_ind:06}.jpg'))

        # s1_ind = np.random.randint(2, vidlen)
        # s0_ind = np.random.randint(1, s1_ind)
        # s2_ind = np.random.randint(s1_ind, vidlen+1)
        # imts0 = self.preprocess(torchvision.io.read_image(f'/datasets01/SSV2/frames/{vid}/{vid}_{s0_ind:06}.jpg'))
        # imts1 = self.preprocess(torchvision.io.read_image(f'/datasets01/SSV2/frames/{vid}/{vid}_{s1_ind:06}.jpg'))
        # imts2 = self.preprocess(torchvision.io.read_image(f'/datasets01/SSV2/frames/{vid}/{vid}_{s2_ind:06}.jpg'))

        label = self.labels[str(vid)]
        return (im0, img, 0, 0, 0, label)

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

        ## Init Model
        self.model = make_network(cfg.agent)
        if cfg.load_snap:
            print("LOADING", cfg.load_snap)
            self.load_snapshot(cfg.load_snap)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

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

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(1000000,
                                       1)
        eval_every_step = utils.Every(5000,
                                      1)

        train_iterable = VideoBuffer(self.cfg.replay_buffer_num_workers, "train", "train", alpha = self.cfg.alpha)
        train_loader = iter(torch.utils.data.DataLoader(train_iterable,
                                         batch_size=self.cfg.batch_size,
                                         num_workers=self.cfg.replay_buffer_num_workers,
                                         pin_memory=True))

        val_iterable = VideoBuffer(self.cfg.replay_buffer_num_workers, "val", "validation", alpha=0)
        val_loader = iter(torch.utils.data.DataLoader(val_iterable,
                                         batch_size=self.cfg.batch_size,
                                         num_workers=self.cfg.replay_buffer_num_workers,
                                         pin_memory=True))
        

        ## Training Loop
        while train_until_step(self.global_step):
            ## Sample Batch
            t0 = time.time()
            batch_frames_0, batch_frames_g, batch_frames_s0, batch_frames_s1, batch_frames_s2, batch_langs = next(train_loader)
            t1 = time.time()

            batch = (batch_frames_0.cuda(), batch_frames_g.cuda(), 
                    batch_frames_s0.cuda(), batch_frames_s1.cuda(), batch_frames_s2.cuda(), batch_langs)
            metrics = self.model.update(batch, self.global_step)
            t2 = time.time()
            self.logger.log_metrics(metrics, self.global_frame, ty='train')
            print(self.global_step, metrics)
            print(f'Sample time {t1-t0}, Update time {t2-t1}')
            
            if eval_every_step(self.global_step):
                with torch.no_grad():
                    batch_frames_0, batch_frames_g, batch_frames_s0, batch_frames_s1, batch_frames_s2, batch_langs = next(val_loader)
                    batch = (batch_frames_0.cuda(), batch_frames_g.cuda(), 
                            batch_frames_s0.cuda(), batch_frames_s1.cuda(), batch_frames_s2.cuda(), batch_langs)
                    metrics = self.model.update(batch, self.global_step, eval=True)
                    self.logger.log_metrics(metrics, self.global_frame, ty='eval')

                    self.save_snapshot()
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.global_step}.pt'
        global_snapshot =  self.work_dir / f'snapshot.pt'
        sdict = {}
        sdict["convnet"] = self.model.encoder.convnet.state_dict()
        sdict["langpred"] = self.model.lang_pred.state_dict()
        torch.save(sdict, snapshot)
        sdict["global_step"] = self._global_step
        torch.save(sdict, global_snapshot)

    def load_snapshot(self, snapshot_path):
        payload = torch.load(snapshot_path)
        self.model.encoder.convnet.load_state_dict(payload['convnet'])
        self.model.lang_pred.load_state_dict(payload['langpred'])
        self._global_step = payload['global_step']

@hydra.main(config_path='cfgs', config_name='config_rep')
def main(cfg):
    from train_representation import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()