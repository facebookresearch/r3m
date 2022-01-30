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
import json
import time
import pickle
from torchvision.utils import save_image
import json
import random


def get_ind(vid, index, ds):
    preprocess = torch.nn.Sequential(
            transforms.Resize(256),
            transforms.CenterCrop(224),
        )
    if ds == "ego4d":
        return torchvision.io.read_image(f"{vid}/{index:06}.jpg")
    elif ds == "sthsth":
        return preprocess(torchvision.io.read_image(f'/datasets01/SSV2/frames/{vid}/{vid}_{index:06}.jpg'))
    elif ds == "robonet":
        return vid[index]


## Data Loader for Ego4D
class R3MBuffer(IterableDataset):
    def __init__(self, num_workers, source1, source2, alpha, datasources, doaug = "none", simclr=False):
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.curr_same = 0
        self.data_sources = datasources
        self.simclr = simclr
        self.doaug = doaug

        # Augmentations
        if self.simclr:
            cj = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            self.aug = torch.nn.Sequential(
                    transforms.RandomResizedCrop(224, scale = (0.08, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([cj], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(23, sigma=(0.1, 2.0))
                )
        else:
            if doaug in ["rc", "rctraj"]:
                # self.aug = torch.nn.Sequential(
                #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                #     transforms.RandomAffine(20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                # )
                self.aug = torch.nn.Sequential(
                    transforms.RandomResizedCrop(224, scale = (0.2, 1.0)),
                )
            else:
                self.aug = lambda a : a

        # Load Data
        if "ego4d" in self.data_sources:
            print("Ego4D")
            self.manifest = pd.read_csv("/private/home/surajn/data/ego4d/manifest.csv")
            print(self.manifest)
            self.ego4dlen = len(self.manifest)
        if "robonet" in self.data_sources:
            print("RoboNet")
            from robonet.datasets import load_metadata, get_dataset_class
            metadata = load_metadata("/private/home/surajn/data/hdf5")
            hparams = {'T': 15, 'action_mismatch': 3, 'state_mismatch': 3, 
                'splits':[0.8, 0.1, 0.1], 'normalize_images':False, 'img_size': [224, 224]}
            self.robonetdata = get_dataset_class("RoboNet")(metadata, source1, hparams)
            self.robonetlen = self.robonetdata.__len__()
        if "sthsth" in self.data_sources:
            self.sthsthdata = pd.read_csv(f"/datasets01/SSV2/{source1}.csv", sep=" ")
            self.sthsthlabels = {}
            f = open(f"/datasets01/SSV2/something-something-v2-{source2}.json")
            for label in json.load(f):
                self.sthsthlabels[label["id"]] = label["label"]


    def _sample(self):
        t0 = time.time()
        ds = random.choice(self.data_sources)

        if ds == "ego4d":
            vidid = np.random.randint(0, self.ego4dlen)
            m = self.manifest.iloc[vidid]
            vidlen = m["len"]
            txt = m["txt"]
            label = txt[2:]
            vid = m["path"]
        elif ds == "sthsth":
            vid = np.random.choice(self.sthsthdata["original_vido_id"])
            vidlen = self.sthsthdata.loc[self.sthsthdata["original_vido_id"] == vid]["frame_id"].max()
            label = self.sthsthlabels[str(vid)]
        elif ds == "robonet":
            vidid = np.random.randint(0, self.robonetlen)
            images, _, _ = self.robonetdata.__getitem__(vidid)
            vid = torch.FloatTensor(images) * 255
            vidlen = 14 
            label = ""

        start_ind = np.random.randint(1, 2 + int(self.alpha * vidlen))
        end_ind = np.random.randint(int((1-self.alpha) * vidlen)-1, vidlen)
        s1_ind = np.random.randint(2, vidlen)
        s0_ind = np.random.randint(1, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen+1)

        if self.simclr:
            im = get_ind(vid, s1_ind, ds)
            imts1_v1 = self.aug(im / 255.0) * 255.0
            imts1_v2 = self.aug(im / 255.0) * 255.0
            return (imts1_v1, imts1_v2)
        else:   
            if self.doaug == "rctraj":
                im0 = get_ind(vid, start_ind, ds) 
                img = get_ind(vid, end_ind, ds)
                imts0 = get_ind(vid, s0_ind, ds)
                imts1 = get_ind(vid, s1_ind, ds)
                imts2 = get_ind(vid, s2_ind, ds)
                allims = torch.stack([im0, img, imts0, imts1, imts2], 0)
                allims_aug = self.aug(allims / 255.0) * 255.0

                im0 = allims_aug[0]
                img = allims_aug[1]
                imts0 = allims_aug[2]
                imts1 = allims_aug[3]
                imts2 = allims_aug[4]
            else:
                im0 = self.aug(get_ind(vid, start_ind, ds) / 255.0) * 255.0
                img = self.aug(get_ind(vid, end_ind, ds) / 255.0) * 255.0
                imts0 = self.aug(get_ind(vid, s0_ind, ds) / 255.0) * 255.0
                imts1 = self.aug(get_ind(vid, s1_ind, ds) / 255.0) * 255.0
                imts2 = self.aug(get_ind(vid, s2_ind, ds) / 255.0) * 255.0

        im = torch.stack([im0, img, imts0, imts1, imts2])
        return (im, label)

    def __iter__(self):
        while True:
            yield self._sample()



## Data Loader for RoboNet
class RoboNetBuffer(IterableDataset):
    def __init__(self, num_workers, source1, source2, alpha, num_same=1, simclr=False):
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.num_same = num_same
        self.curr_same = 0

        self.simclr = simclr

        # Augmentations
        if self.simclr:
            cj = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            self.aug = torch.nn.Sequential(
                    transforms.RandomResizedCrop(224, scale = (0.08, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([cj], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(23, sigma=(0.1, 2.0))
                )
        else:
            self.aug = torch.nn.Sequential(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.RandomAffine(20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            )


        # Load Data
        print("RoboNet")
        from robonet.datasets import load_metadata, get_dataset_class
        print("W")
        metadata = load_metadata("/private/home/surajn/data/hdf5")
        print("A")
        hparams = {'T': 15, 'action_mismatch': 3, 'state_mismatch': 3, 
            'splits':[0.8, 0.1, 0.1], 'normalize_images':False, 'img_size': [224, 224]}
        self.dataset = get_dataset_class("RoboNet")(metadata, "train", hparams)
        print("D")
        self.len = self.dataset.__len__()
        print(self.len)
        # idx = np.random.randint(0, self.len)

        # video, _, _ = self.dataset.__getitem__(idx)
        # print(video.shape)
        # self.loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16)


        # images, states, actions = next(iter(loader))
        # images = np.transpose(images.numpy(), (0, 1, 3, 4, 2))
        # images *= 255
        # images = images.astype(np.uint8)

        # print(images.shape)
        # import imageio
        # writer = imageio.get_writer('test_frames.gif')
        # for t in range(images.shape[1]):
        #     writer.append_data(np.concatenate([b for b in images[:, t]], axis=-2))
        # writer.close()           

    def _sample(self):
        t0 = time.time()
        idx = np.random.randint(0, self.len)

        images, _, _ = self.dataset.__getitem__(idx)
        images = torch.FloatTensor(images) * 255
        vidlen = 14 #m["len"]

        # if self.simclr:
        #     s1_ind = np.random.randint(2, vidlen)
        #     imts1_v1 = self.aug(torchvision.io.read_image(f"{path}/{s1_ind:06}.jpg") / 255.0) * 255.0
        #     imts1_v2 = self.aug(torchvision.io.read_image(f"{path}/{s1_ind:06}.jpg") / 255.0) * 255.0
        #     return (imts1_v1, imts1_v2)

        start_ind = np.random.randint(1, 2 + int(self.alpha * vidlen))
        end_ind = np.random.randint(int((1-self.alpha) * vidlen)-1, vidlen)
        im0 = self.aug(images[start_ind] / 255.0) * 255.0
        img = self.aug(images[end_ind] / 255.0) * 255.0

        s1_ind = np.random.randint(2, vidlen)
        s0_ind = np.random.randint(1, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen+1)
        imts0 = self.aug(images[s0_ind] / 255.0) * 255.0
        imts1 = self.aug(images[s1_ind] / 255.0) * 255.0
        imts2 = self.aug(images[s2_ind] / 255.0) * 255.0

        im = torch.stack([im0, img, imts0, imts1, imts2])
        return im, ""

    def __iter__(self):
        while True:
            yield self._sample()

## Data Loader for Ego4D
class Ego4DBuffer(IterableDataset):
    def __init__(self, num_workers, source1, source2, alpha, num_same=1, simclr=False):
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.num_same = num_same
        self.curr_same = 0

        self.simclr = simclr

        # Augmentations
        if self.simclr:
            cj = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            self.aug = torch.nn.Sequential(
                    transforms.RandomResizedCrop(224, scale = (0.08, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([cj], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(23, sigma=(0.1, 2.0))
                )
        else:
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
        if self.curr_same % self.num_same == 0:
            parentvid = np.random.randint(0, self.mlen)
            parentm = self.manifest.iloc[parentvid]
            path = parentm["path"]

            vidsp = path.split("/")[:-2]
            self.vidsp = "/".join(vidsp)

            scl = [self.vidsp in p for p in self.manifest["path"]]
            self.num_subclips = np.sum(scl)
            self.subclips = self.manifest.loc[scl]

        vid = np.random.randint(0, self.num_subclips)
        m = self.subclips.iloc[vid]
        vidlen = m["len"]
        txt = m["txt"]
        txt = txt[2:]
        path = m["path"]

        if self.simclr:
            s1_ind = np.random.randint(2, vidlen)
            imts1_v1 = self.aug(torchvision.io.read_image(f"{path}/{s1_ind:06}.jpg") / 255.0) * 255.0
            imts1_v2 = self.aug(torchvision.io.read_image(f"{path}/{s1_ind:06}.jpg") / 255.0) * 255.0
            return (imts1_v1, imts1_v2)

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

        self.curr_same += 1
        label = txt
        im = torch.stack([im0, img, imts0, imts1, imts2])
        return (im, label)

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
                    path = f"/private/home/surajn/code/vrl_private/vrl/hydra/expert_data/final_paths_multiview_rb_200/{camera}/{t}.pickle"
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

