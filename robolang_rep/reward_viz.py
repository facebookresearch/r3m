

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
# from robolang_rep.data_loaders import R3MBuffer, RoboNetBuffer, Ego4DBuffer, GTBuffer, VideoBuffer
from logger import Logger
import json
import time
import pickle
from torchvision.utils import save_image
import json
import random
import av
import copy

from pytorch_grad_cam import GradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2

torch.backends.cudnn.benchmark = True


def make_network(cfg):
    model =  hydra.utils.instantiate(cfg)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)
    return model.cuda()


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
        vidlen = 0
        while not ((vidlen > 80) and (vidlen < 100)):
            vid = np.random.randint(0, self.mlen)
            m = self.manifest.iloc[vid]
            vidlen = m["len"]
        txt = m["txt"]
        txt = txt[2:]
        path = m["path"]

        ims = []
        for i in range(80):
            ims.append(torchvision.io.read_image(f"{path}/{i:06}.jpg"))
        ims = torch.stack(ims)
        label = txt
        return (ims, label)

    def __iter__(self):
        while True:
            yield self._sample()

## Data Loader for Ground Truth Franka Kitchen Demo Data
class MWBuffer(IterableDataset):
    def __init__(self, num_workers, source1, source2, alpha, shuf, gt=False, alldata=None, view=None, task=None):
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
        self.cams = ["top_cap2", "left_cap2", "right_cap2"] #[view] #
        self.tasks = ["assembly-v2-goal-observable","bin-picking-v2-goal-observable","button-press-topdown-v2-goal-observable",
                "drawer-open-v2-goal-observable","hammer-v2-goal-observable"] #,"kitchen_rdoor_open-v3"] [task] #
        self.instr = ["Inserting the ring into the peg", "Picking the block and placing it into the other bin", 
                "Pressing the red button",  "Opening the drawer", 
                "Hammering the nail into the wood"] #, "Opening the middle door"]
        if alldata is not None:
            self.all_demos, self.all_labels, self.all_states = alldata
        else:
            all_demos = []
            all_labels = []
            all_states = []
            for camera in self.cams:
                for i, t in enumerate(self.tasks):
                    path = f"/private/home/surajn/code/vrl_private/vrl/hydra/expert_data/final_paths_multiview_meta_200/{camera}/{t}.pickle"
                    demo_paths = pickle.load(open(path, 'rb'))
                    demo_paths = demo_paths[:1000]
                    print(len(demo_paths), i, t)
                    for p in demo_paths:
                        all_demos.append(p["images"][::10])
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

        video_p = self.preprocess(video / 255.0 ) * 255.0
        label = self.all_labels[vid]
        return (video_p, label)

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

## Data Loader for Ground Truth Franka Kitchen Demo Data
class GTBuffer(IterableDataset):
    def __init__(self, num_workers, source1, source2, alpha, shuf, gt=False, alldata=None, view=None, task=None):
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
        self.cams = ["default", "left_cap2", "right_cap2"] #[view] 
        self.tasks = ["kitchen_knob1_on-v3","kitchen_light_on-v3","kitchen_sdoor_open-v3","kitchen_micro_open-v3","kitchen_ldoor_open-v3"] #[task] 
        self.instr = ["Turning the bottom right knob clockwise", "Switching the light on", 
                "Sliding the top right door open", "Opening the microwave", 
                "Opening the top left door"] #, "Opening the middle door"]
        if alldata is not None:
            self.all_demos, self.all_labels, self.all_states = alldata
        else:
            all_demos = []
            all_labels = []
            all_states = []
            for camera in self.cams:
                for i, t in enumerate(self.tasks):
                    path = f"/private/home/surajn/code/vrl_private/vrl/hydra/expert_data/final_paths_multiview_rb_200/{camera}/{t}.pickle"
                    demo_paths = pickle.load(open(path, 'rb'))
                    demo_paths = demo_paths[:200]
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

        video_p = self.preprocess(video / 255.0 ) * 255.0
        label = self.all_labels[vid]
        return (video_p, label)

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
            # shuf = np.random.choice(200*3, 200*3, replace=False)
            train_iterable = GTBuffer(self.cfg.replay_buffer_num_workers, "train", "train",
                     alpha = self.cfg.alpha, shuf = shuf, gt = 0, view=self.cfg.view, task=self.cfg.task)
            alldata = (train_iterable.all_demos, train_iterable.all_labels, train_iterable.all_states)
            val_iterable = GTBuffer(self.cfg.replay_buffer_num_workers, "val", "validation",
                     alpha=0, shuf = shuf, gt = 0, alldata=alldata, view=self.cfg.view, task=self.cfg.task)
        elif self.cfg.dataset == "mw":
            shuf = np.random.choice(200*5*3, 200*5*3, replace=False)
            # shuf = np.random.choice(200, 200, replace=False)
            train_iterable = MWBuffer(self.cfg.replay_buffer_num_workers, "train", "train",
                     alpha = self.cfg.alpha, shuf = shuf, gt = 0, view=self.cfg.view, task=self.cfg.task)
            alldata = (train_iterable.all_demos, train_iterable.all_labels, train_iterable.all_states)
            val_iterable = MWBuffer(self.cfg.replay_buffer_num_workers, "val", "validation",
                     alpha=0, shuf = shuf, gt = 0, alldata=alldata, view=self.cfg.view, task=self.cfg.task)

        self.train_loader = iter(torch.utils.data.DataLoader(train_iterable,
                                         batch_size=self.cfg.batch_size,
                                         num_workers=self.cfg.replay_buffer_num_workers,
                                         pin_memory=True))
        self.val_loader = iter(torch.utils.data.DataLoader(val_iterable,
                                         batch_size=self.cfg.batch_size,
                                         num_workers=self.cfg.replay_buffer_num_workers,
                                         pin_memory=True))
        self.instr = val_iterable.instr



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


    def plot_goal_dist(self, goal, demos, model, langs):
        rs= []
        rns = []
        pixrs= []
        pixrns = []
        cliprs= []
        cliprns = []
        maps = {}
        import clip
        clipmodel, _ = clip.load("RN50", device="cuda")
        import torchvision.transforms as T
        # cliptransforms = T.Compose([T.Resize(224),
        #                         T.CenterCrop(224),
        #                         T.ToTensor(), # ToTensor() divides by 255
        #                         T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])])

        cliptransforms = T.Compose([T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])])

        print(goal.shape, demos.shape)
        print(goal.min(), goal.max(), demos.min(), demos.max())
        for ts in range(0, demos.shape[1]):
            print(ts)
            metric = {}
            with torch.no_grad():
                img = demos[:, -1] #goal.unsqueeze(0).repeat(demos.shape[0], 1, 1, 1)
                imt = demos[:, ts]
                imn = demos[:, ts]
                eg = self.model(img)[0]
                et = self.model(imt)[0]
                en = self.model(imn)[0]
                r = -torch.linalg.norm((eg-et), dim = -1)
                r_n = -torch.linalg.norm((eg-en), dim = -1)

                clipeg = clipmodel.encode_image(cliptransforms(img / 255.0))
                clipet = clipmodel.encode_image(cliptransforms(imt / 255.0))
                clipen = clipmodel.encode_image(cliptransforms(imn / 255.0))
                # print(clipen.shape)
                # print(clipen)
                clipr = -torch.linalg.norm((clipeg-clipet), dim = -1)
                clipr_n = -torch.linalg.norm((clipeg-clipen), dim = -1)

                pixr = -torch.linalg.norm((img-imt), dim = ((1,2,3)))
                pixr_n = -torch.linalg.norm((imt-imn), dim = ((1,2,3)))

                rs.append(r.cpu().detach().numpy())
                rns.append(r_n.cpu().detach().numpy())
                
                pixrs.append(pixr.cpu().detach().numpy())
                pixrns.append(pixr_n.cpu().detach().numpy())

                cliprs.append(clipr.cpu().detach().numpy())
                cliprns.append(clipr_n.cpu().detach().numpy())
        rs = np.stack(rs, -1)
        rs = rs - rs.min()
        rs = rs / rs.max()
        rse = rs.std(0) / np.sqrt(rs.shape[0])
        rs = rs.mean(0)
        # rns = np.stack(rns, -1)
        pixrs = np.stack(pixrs, -1)
        pixrs = pixrs - pixrs.min()
        pixrs = pixrs / pixrs.max()
        pixrse = pixrs.std(0) / np.sqrt(pixrs.shape[0])
        pixrs = pixrs.mean(0)

        cliprs = np.stack(cliprs, -1)
        cliprs = cliprs - cliprs.min()
        cliprs = cliprs / cliprs.max()
        cliprse = cliprs.std(0) / np.sqrt(cliprs.shape[0])
        cliprs = cliprs.mean(0)
        # pixrns = np.stack(pixrns, -1)
        print(rs.shape, pixrs.shape, cliprs.shape)

        for b in range(3):
            filename = self.work_dir.joinpath(f"{b}_dem_{self.global_step}.gif")
            from moviepy.editor import ImageSequenceClip
            clip = ImageSequenceClip(list(demos[b].permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)), fps=20)
            clip.write_gif(filename, fps=20)
            import matplotlib.pyplot as plt
            
        plt.plot(rs, label=f"R3M", color="red")
        plt.fill_between(range(0, 50), rs-rse, rs+rse, color="red", alpha=0.1)
        plt.plot(pixrs,  label=f"Pixel", color="blue")
        plt.fill_between(range(0, 50), pixrs-pixrse, pixrs+pixrse, color="blue", alpha=0.1)
        plt.plot(cliprs,  label=f"CLIP", color="green")
        plt.fill_between(range(0, 50), cliprs-cliprse, cliprs+cliprse, color="green", alpha=0.1)
        # plt.plot(cliprs[b], label=f"True Demo {batch_langs[b]} (CLIP)")
        plt.legend()
        plt.savefig(self.work_dir.joinpath(f"{b}_rew_{self.global_step}.png"))
        plt.close()
        assert(False)

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
        batch_vids, batch_langs = next(self.val_loader)
        # batch_langs_shuf1 = copy.deepcopy(batch_langs)
        # random.shuffle(batch_langs_shuf1)
        # batch_langs_shuf2 = copy.deepcopy(batch_langs)
        # random.shuffle(batch_langs_shuf2)
        # batch_langs_shuf3 = copy.deepcopy(batch_langs)
        # random.shuffle(batch_langs_shuf3)
        batch_vids = batch_vids.cuda()

        # batch_vids_shuf = []
        # for i in range(batch_vids.shape[0]):
        #     ranid = i
        #     while batch_langs[ranid] == batch_langs[i]:
        #         ranid = np.random.randint(0, batch_vids.shape[0])
        #     batch_vids_shuf.append(batch_vids[ranid])
        # batch_vids_shuf = torch.stack(batch_vids_shuf, 0)

        t1 = time.time()
        self.model.eval()

        self.plot_goal_dist(batch_vids[0, -1], batch_vids[1:], self.model, batch_langs)
        assert(False)

        rs= []
        rns = []
        pixrs= []
        pixrns = []
        cliprs= []
        cliprns = []
        maps = {}

        # for iss in self.instr:
        #     print(iss)
        #     r, _ = self.model.module.get_reward(self.model(batch_vids[:, 5])[0], self.model(batch_vids[:, -5])[0], [iss]*batch_vids.shape[0])
        #     print(r.shape)
        #     for b in range(batch_vids.shape[0]):
        #         self.work_dir.joinpath(f"{iss}").mkdir(parents=True, exist_ok=True)
        #         filename = self.work_dir.joinpath(f"{iss}/score_{r[b]:.6}.gif")
        #         from moviepy.editor import ImageSequenceClip
        #         clip = ImageSequenceClip(list(batch_vids[b].permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)), fps=20)
        #         clip.write_gif(filename, fps=20)

        # assert(False)

        import clip
        clipmodel, _ = clip.load("RN50", device="cuda")
        import torchvision.transforms as T
        cliptransforms = T.Compose([T.Resize(224),
                                T.CenterCrop(224),
                                T.ToTensor(), # ToTensor() divides by 255
                                T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])])

        for ts in range(0, batch_vids.shape[1]):
            print(ts)
            metric = {}
            # with torch.no_grad():
            #     r, a = self.model.module.get_reward(self.model(batch_vids[:, 0])[0], self.model(batch_vids[:, ts])[0], batch_langs)
            #     # r_n1, an = self.model.module.get_reward(self.model(batch_vids[:, 0])[0], self.model(batch_vids[:, ts])[0], batch_langs_shuf1)
            #     # r_n2, an = self.model.module.get_reward(self.model(batch_vids[:, 0])[0], self.model(batch_vids[:, ts])[0], batch_langs_shuf2)
            #     # r_n3, an = self.model.module.get_reward(self.model(batch_vids[:, 0])[0], self.model(batch_vids[:, ts])[0], batch_langs_shuf3)
            #     r_n4, an = self.model.module.get_reward(self.model(batch_vids_shuf[:, 0])[0], self.model(batch_vids_shuf[:, ts])[0], batch_langs)

            with torch.no_grad():
                img = batch_vids[:, -1]
                imt = batch_vids[:, ts]
                imn = batch_vids_shuf[:, ts]
                eg = self.model(img)[0]
                et = self.model(imt)[0]
                en = self.model(imn)[0]
                r = -torch.linalg.norm((eg-et), dim = -1)
                r_n4 = -torch.linalg.norm((eg-en), dim = -1)

                clipeg = clipmodel.encode_image(cliptransforms(img))
                clipet = clipmodel.encode_image(cliptransforms(imt))
                clipen = clipmodel.encode_image(cliptransforms(imn))
                print(clipen.shape)
                print(clipen)
                clipr = -torch.linalg.norm((clipeg-clipet), dim = -1)
                clipr_n4 = -torch.linalg.norm((clipeg-clipen), dim = -1)

                pixr = -torch.linalg.norm((img-imt), dim = ((1,2,3)))
                pixr_n4 = -torch.linalg.norm((imt-imn), dim = ((1,2,3)))
                # r, a = self.model.module.get_reward(self.model(batch_vids[:, 0])[0], self.model(batch_vids[:, ts])[0], batch_langs)
                # r_n4, an = self.model.module.get_reward(self.model(batch_vids_shuf[:, 0])[0], self.model(batch_vids_shuf[:, ts])[0], batch_langs)

            target_layers = [self.model.module.convnet.layer1[0]]
            with GradCAM(model=self.model.module.convnet,
                target_layers=target_layers,
                use_cuda=1) as cam:


                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = batch_vids[:, 0].shape[0]
                grayscale_cam = cam(input_tensor=batch_vids[:, ts],
                                    targets=None,
                                    aug_smooth=0,
                                    eigen_smooth=0)

                for i in range(batch_vids.shape[0]):
                    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                    im  = batch_vids[i, ts].permute(1, 2, 0).cpu().detach().numpy() / 255.0
                    cam_image = show_cam_on_image(im, grayscale_cam[i, :], use_rgb=True)
                    c = cam_image.astype(np.uint8)

                    try:
                        maps[i].append(c)
                    except:
                        maps[i] = [c]
                    
                    # dirname = self.work_dir.joinpath(f"cam_{i}")
                    # os.makedirs(dirname, exist_ok=True)
                    # cv2.imwrite(f'{dirname}/{ts}.jpg', c)
                

            r_n = r_n4 # torch.stack([r_n1, r_n2, r_n3], -1)
            print(r, pixr, clipr)
            metric["rew_pos"] = r.mean()
            metric["rew_neg"] = r_n.mean()
            rs.append(r.cpu().detach().numpy())
            rns.append(r_n.cpu().detach().numpy())
            
            pixrs.append(pixr.cpu().detach().numpy())
            pixrns.append(pixr_n4.cpu().detach().numpy())

            cliprs.append(clipr.cpu().detach().numpy())
            cliprns.append(clipr_n4.cpu().detach().numpy())
            self.logger.log_metrics(metric, ts, ty='train')
        rs = np.stack(rs, -1)
        rs = rs - rs.min()
        rs = rs / rs.max()
        # rns = np.stack(rns, -1)
        pixrs = np.stack(pixrs, -1)
        pixrs = pixrs - pixrs.min()
        pixrs = pixrs / pixrs.max()

        cliprs = np.stack(cliprs, -1)
        cliprs = cliprs - cliprs.min()
        cliprs = cliprs / cliprs.max()
        # pixrns = np.stack(pixrns, -1)

        for b in range(batch_vids.shape[0]):
            filename = self.work_dir.joinpath(f"{b}_dem_{self.global_step}.gif")
            from moviepy.editor import ImageSequenceClip
            clip = ImageSequenceClip(list(batch_vids[b].permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)), fps=20)
            clip.write_gif(filename, fps=20)

            filename = self.work_dir.joinpath(f"{b}_att_{self.global_step}.gif")
            clipmap = ImageSequenceClip(maps[b], fps=20)
            clipmap.write_gif(filename, fps=20)

            import matplotlib.pyplot as plt
            
            # plt.plot(rns[b], label=batch_langs_shuf1[b])
            # plt.plot(rns[b, 1], label=batch_langs_shuf2[b])
            # plt.plot(rns[b, 2], label=batch_langs_shuf3[b])
            plt.plot(rs[b], label=f"True Demo {batch_langs[b]} (R3M)")
            # plt.plot(rns[b], label="Different Demo (R3M)")
            plt.plot(pixrs[b], label=f"True Demo {batch_langs[b]} (Pixel)")

            plt.plot(cliprs[b], label=f"True Demo {batch_langs[b]} (CLIP)")
            # plt.plot(pixrns[b], label="Different Demo (Pixel)")
            plt.legend()
            plt.savefig(self.work_dir.joinpath(f"{b}_rew_{self.global_step}.png"))
            plt.close()
        self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.global_step}.pt'
        global_snapshot = self.work_dir / f'snapshot.pt'
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

@hydra.main(config_path='cfgs', config_name='config_rew_viz')
def main(cfg):
    from reward_viz import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot(snapshot)
    workspace.train()


if __name__ == '__main__':
    main()