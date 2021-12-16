# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Identity
import torchvision
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel, AutoConfig
from robolang_rep import utils
from pathlib import Path
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from robolang_rep.models_language import LangEncoder, LanguageReward, LanguageAttention

epsilon = 1e-8
def do_nothing(x): return x

class R3M(nn.Module):
    def __init__(self, device, lr, hidden_dim, finetune=1, pretrained=0, size=34, l2weight=1.0, l1weight=1.0, 
                 langweight=1.0, tcnweight=0.0, structured=False, lang_cond=False, 
                 l2dist=True, attntype="modulate", finetunelang=0,
                 cpcweight = 0.0, num_same=1, langtype="reconstruct", anneall1=False, mask=True):
        super().__init__()

        self.device = device
        self.use_tb = False
        self.l2weight = l2weight
        self.l1weight = l1weight
        self.anneall1 = anneall1 ## Anneal up L1 Penalty
        self.lang_cond = lang_cond ## Language conditioned or not
        self.tcnweight = tcnweight ## Weight on TCN loss (states closer in same clip closer in embedding)
        self.cpcweight = cpcweight ## Weight on CPC loss (states within same clip closer than other clips)
        self.num_same = num_same ## How many clips from same scene per batch
        self.finetunelang = finetunelang ## Finetune language model
        self.l2dist = l2dist ## Use -l2 or cosine sim
        self.mask = mask ## Learn Binary mask over image features
        self.attntype = attntype ## Type of language-conditioned attention
        self.langtype = langtype ## Type of language based reward
        self.langweight = langweight ## Weight on language reward
        self.size = size ## Size ResNet or ViT
        self.finetune = finetune ## Train Model
        self.proprio_shape = 0 ## Amount of proprioceptive features at front of observation

        ## Distances and Metrics
        self.cs = torch.nn.CosineSimilarity(1)
        self.bce = nn.BCELoss(reduce=False)
        self.sigm = Sigmoid()

        params = []
        ######################################################################## Sub Modules
        ## Pretrained DistilBERT Sentence Encoder
        self.lang_enc = LangEncoder(self.finetunelang, 0) 
        if self.finetunelang:
            params += list(self.lang_enc.parameters())

        ## Visual Encoder
        if size == 18:
            self.outdim = 512
            self.convnet = torchvision.models.resnet18(pretrained=pretrained)
        elif size == 34:
            self.outdim = 512
            self.convnet = torchvision.models.resnet34(pretrained=pretrained)
        elif size == 50:
            self.outdim = 2048
            self.convnet = torchvision.models.resnet50(pretrained=pretrained)
        elif size == 0:
            self.outdim = 768
            if pretrained:
                self.convnet = AutoModel.from_pretrained('google/vit-base-patch32-224-in21k').to('cuda')
            else:
                self.convnet = AutoModel.from_config(config = AutoConfig.from_pretrained('google/vit-base-patch32-224-in21k')).to('cuda')
        if self.size == 0:
            self.normlayer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            self.normlayer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.convnet.fc = Identity()
        if self.finetune:
            self.convnet.train()
        else:
            self.convnet.eval()
        params += list(self.convnet.parameters())
        
        ## IF REPRESENTATION IS LANGUAGE CONDITIONED
        if self.lang_cond:
            self.lang_attn = LanguageAttention(self.attntype, self.outdim, self.lang_enc.lang_size)
            params += list(self.lang_attn.parameters())

        ## Language Reward
        if self.langweight > 0.0:
            self.lang_rew = LanguageReward(self.langtype, self.outdim, hidden_dim, self.lang_enc.lang_size, simfunc=self.sim) 
            params += list(self.lang_rew.parameters())

        ## Sparsity Mask
        if self.mask:
            self.m = torch.rand(self.outdim, requires_grad=True, device="cuda")
            params += list([self.m])
        else:
            self.m = None
        ########################################################################

        ## Optimizer
        self.encoder_opt = torch.optim.Adam(params, lr = lr)


    def encode(self, image, sentences = None):
        ## Assumes preprocessing and resizing is done
        e = self.convnet(image)
        if self.lang_cond:
            le = self.lang_enc(sentences)
            e, a = self.lang_attn(e, le)

        if self.mask:
            a = self.sigm(self.m.unsqueeze(0).repeat(e.shape[0], 1))
            e = e * a.to(e.device)
        return e, a

    def get_reward(self, e0, es, sentences):
        le = self.lang_enc(sentences)
        return self.lang_rew(e0, es, le)

    ## Forward Call (im --> representation)
    def forward(self, obs, sentences = None, num_ims = 1, obs_shape = [3, 224, 224]):
        ## If proprioceptive data is stacked at front of obs
        if self.proprio_shape > 0:
            proprio = obs[:, -self.proprio_shape:]
            obs = obs[:, :-self.proprio_shape].reshape([obs.shape[0]] + list(obs_shape))

        if obs_shape != [3, 224, 224]:
            preprocess = nn.Sequential(
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        self.normlayer,
                )
        else:
            preprocess = nn.Sequential(
                        self.normlayer,
                )

        ## Input must be [0, 255]
        obs = obs.float() /  255.0
        h = []
        for i in range(0, num_ims*3, 3):
            obs_p = preprocess(obs[:, i:(i+3)])
            if self.finetune:
                e, a = self.encode(obs_p, sentences)
            else:
                with torch.no_grad():
                    e, a = self.convnet(obs_p, sentences)
            h.append(e)
        h = torch.cat(h, -1)
        h = h.view(h.shape[0], -1)

        ## Add back proprioception if there
        if self.proprio_shape > 0:
            h = torch.cat([h, proprio], -1)
        return (h, a)

    def sim(self, tensor1, tensor2):
        if self.l2dist:
            d = - torch.linalg.norm(tensor1 - tensor2, dim = -1) #torch.sqrt(((es1 - es0)**2).mean(-1))
        else:
            d = self.cs(tensor1, tensor2)
        return d
