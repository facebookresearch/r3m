# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Identity
import torchvision
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel, AutoConfig
import utils
import random
from pathlib import Path
from torchvision.utils import save_image

import matplotlib.pyplot as plt


def do_nothing(x): return x

class ResnetEncoder(nn.Module):
    def __init__(self, obs_shape, finetune, pretrained, reshape=0, num_ims=3, size=18, proprio_shape=0):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_ims = num_ims
        self.obs_shape = obs_shape
        self.proprio_shape = proprio_shape
        if size in [18, 34]:
            self.output_dim = 512
        elif size == 50:
            self.output_dim = 2048
        self.repr_dim = self.output_dim * num_ims + proprio_shape
        self.finetune = finetune
        if reshape:
            self.preprocess = nn.Sequential(
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                )
        else:
            self.preprocess = nn.Sequential(
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                )
        
        if size == 18:
            self.convnet = torchvision.models.resnet18(pretrained=pretrained)
        elif size == 34:
            self.convnet = torchvision.models.resnet34(pretrained=pretrained)
        elif size == 50:
            self.convnet = torchvision.models.resnet50(pretrained=pretrained)
        self.convnet.fc = Identity()
        if self.finetune:
            self.convnet.train()
        else:
            self.convnet.eval()

        if not pretrained:
            self.apply(utils.weight_init)

    def forward(self, obs):
        if self.proprio_shape > 0:
            proprio = obs[:, -self.proprio_shape:]
            obs = obs[:, :-self.proprio_shape].reshape([obs.shape[0]] + list(self.obs_shape))
        obs = obs / 255.0
        h = []
        for i in range(0, self.num_ims*3, 3):
            obs_p = self.preprocess(obs[:, i:(i+3)])
            if self.finetune:
                e = self.convnet(obs_p).squeeze(-1).squeeze(-1)
            else:
                with torch.no_grad():
                    e = self.convnet(obs_p).squeeze(-1).squeeze(-1)
            h.append(e)
        h = torch.cat(h, -1)
        h = h.view(h.shape[0], -1)
        if self.proprio_shape > 0:
            h = torch.cat([h, proprio], -1)
        return h

class Representation:
    def __init__(self, device, lr, feature_dim,
                 hidden_dim, use_tb, finetune=1, pretrained=0, size=34, l2weight=1.0, smoothweight=0.0, structured=False):

        self.device = device
        self.use_tb = use_tb
        self.l2weight=l2weight
        self.smoothweight = smoothweight
        self.encoder = ResnetEncoder((224, 224, 3), finetune, pretrained, reshape=0, num_ims=1, size=size).to(device)
        self.lang_pred = LangPredictor(self.encoder.repr_dim, hidden_dim, structured=structured).to(device)
        self.repr_dim = self.encoder.repr_dim
        self.aug = torch.nn.Sequential(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.RandomAffine(20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            ).cuda()

        # optimizers
        self.encoder_opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.lang_pred.parameters()), lr=lr)

    def log_batch(self, b_im0, b_img, step, lang, eval):
        im_logging = []
        for i in range(10):
            im = torch.cat([b_im0[i], b_img[i]], 1)
            im_logging.append(im)
        ims_log = torch.cat(im_logging, -1) / 255.0
        work_dir = Path.cwd().joinpath('ims')
        work_dir.mkdir(parents=True, exist_ok=True)
        save_image(ims_log, work_dir.joinpath(f"{eval}_im_{step}.png"))
        with open(work_dir.joinpath(f"{eval}_lang_{step}.txt"), 'w') as f:
            for item in lang[:10]:
                f.write("%s\n" % item)

        #     true_preds = torch.diagonal(preds).permute(1,0)
        #     false_preds1 = preds[:, 0]
        #     false_preds2 = preds[:, 1]
        #     false_preds3 = preds[:, 2]
        #     al = torch.cat([e0, eg, true_preds, false_preds1, false_preds2, false_preds3], 0)
        #     print(al.shape)
        #     u, s, v = torch.pca_lowrank(al, q=2, niter=100)
        #     print(u.shape, s.shape, v.shape)
        #     u = u.cpu().detach().numpy()
        #     plt.scatter(u[0:bs, 0], u[0:bs, 1], label="e0")
        #     plt.scatter(u[bs:(2*bs), 0], u[bs:(2*bs), 1], label="eg")
        #     plt.scatter(u[(2*bs):(3*bs), 0], u[(2*bs):(3*bs), 1], label="tp")
        #     plt.scatter(u[(3*bs):(4*bs), 0], u[(3*bs):(4*bs), 1], label="fp1")
        #     plt.scatter(u[(4*bs):(5*bs), 0], u[(4*bs):(5*bs), 1], label="fp2")
        #     plt.scatter(u[(5*bs):(6*bs), 0], u[(5*bs):(6*bs), 1], label="fp3")
        #     plt.legend()
        #     plt.savefig("test.png")

    def update(self, batch, step, eval=False):
        metrics = dict()
        if eval:
            self.encoder.eval()
            self.lang_pred.eval()
        else:
            self.encoder.train()
            self.lang_pred.train()

        ## Batch and Augment
        b_im0, b_img, b_s0, b_s1, b_s2, b_lang = batch
        if not eval:
            b_im0 = self.aug(b_im0.float() / 255.0) * 255
            b_img = self.aug(b_img.float() / 255.0) * 255

        ## Encode Start and End Frames
        bs = b_im0.shape[0]
        e0 = self.encoder(b_im0)
        eg = self.encoder(b_img)

        ## Do language prediction for each instruction
        e0r = e0.repeat(bs, 1)
        langsr = np.repeat(b_lang, bs)
        preds = self.lang_pred(e0r, langsr).reshape((bs, bs, self.repr_dim))

        ## Compute CPC Loss
        emb_g = eg.unsqueeze(1).repeat(1, bs, 1)
        sims = - torch.sqrt(((preds - emb_g)**2).mean(-1))
        sims_e = torch.exp(sims)
        loss_neglang = -torch.log(torch.diagonal(sims_e) / sims_e.sum(1)).mean()
        l2loss = (torch.linalg.norm(e0) + torch.linalg.norm(eg)) / 2.0

        ## Compute initial/final CPC Loss
        true_preds = torch.diagonal(preds).permute(1,0)
        sim_g = - torch.sqrt(((true_preds - eg)**2).mean(-1))
        sim_0 = - torch.sqrt(((true_preds - e0)**2).mean(-1))
        smoothloss = -torch.log(torch.exp(sim_g) / (torch.exp(sim_0) + torch.exp(sim_g))).mean()

        true_cls = torch.LongTensor(range(0 , bs)).cuda()
        a = utils.accuracy(sims_e, true_cls, (1, 5))
        a_state = ((sim_0 < sim_g)*1.0).mean()

        metrics['loss_lang'] = loss_neglang.item()
        # metrics['loss_state'] = loss_negstate.item()
        metrics['l2loss'] = l2loss.item()
        metrics['smoothloss'] = smoothloss.item()
        metrics['numerator'] = torch.diagonal(sims_e).mean().item()
        metrics['lang_accuracy1'] = a[0]
        metrics['lang_accuracy5'] = a[1]
        metrics['state_accuracy1'] = a_state
        metrics['denomerator'] = sims_e.sum(1).mean().item()
        
        if not eval:
            self.encoder_opt.zero_grad()
            (loss_neglang + self.l2weight * l2loss + self.smoothweight * smoothloss).backward()
            self.encoder_opt.step()

        if step % 5000 == 0:
            self.log_batch(b_im0, b_img, step, b_lang, eval)

        return metrics


class LangEncoder(nn.Module):
  def __init__(self, finetune = False, scratch=False):
    super().__init__()
    self.finetune = finetune
    self.scratch = scratch # train from scratch vs load weights
    self.device = "cuda"
    self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if not self.scratch:
      self.model = AutoModel.from_pretrained("distilbert-base-uncased").to('cuda')
    else:
      self.model = AutoModel.from_config(config = AutoConfig.from_pretrained("distilbert-base-uncased")).to('cuda')
    self.lang_size = 768
      
  def forward(self, langs):
    try:
      langs = langs.tolist()
    except:
      pass
    
    if self.finetune:
      encoded_input = self.tokenizer(langs, return_tensors='pt', padding=True)
      input_ids = encoded_input['input_ids'].to(self.device)
      attention_mask = encoded_input['attention_mask'].to(self.device)
      lang_embedding = self.model(input_ids, attention_mask=attention_mask)[0][:, -1]
    else:
      with torch.no_grad():
        encoded_input = self.tokenizer(langs, return_tensors='pt', padding=True)
        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)
        lang_embedding = self.model(input_ids, attention_mask=attention_mask)[0][:, -1]
    return lang_embedding

class LangPredictor(nn.Module):
    def __init__(self, feature_dim, hidden_dim, structured=False):
        super().__init__()
        self.lang_enc = LangEncoder(0, 0)
        self.structured = structured
        print(feature_dim+self.lang_enc.lang_size)
        if structured:
            self.pred = nn.Sequential(nn.Linear(self.lang_enc.lang_size, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, feature_dim))
        else:
            self.pred = nn.Sequential(nn.Linear(feature_dim+self.lang_enc.lang_size, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, feature_dim))
        
    def forward(self, emb, lang):
        if self.structured:
            lang_emb = self.pred(self.lang_enc(lang))
            emb_pred = emb + lang_emb
        else:
            lang_emb = self.lang_enc(lang)
            embin = torch.cat([emb, lang_emb], -1)
            emb_pred = self.pred(embin)
        return emb_pred
