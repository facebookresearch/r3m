# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from pathlib import Path
from torchvision.utils import save_image
import time
import copy
import matplotlib.pyplot as plt
import torchvision.transforms as T

epsilon = 1e-8
def do_nothing(x): return x

class Trainer():
    def __init__(self, eval_freq):
        self.eval_freq = eval_freq

    def update(self, model, batch, step, eval=False):
        t0 = time.time()
        metrics = dict()
        if eval:
            model.eval()
        else:
            model.train()

        t1 = time.time()
        ## Batch
        b_im, b_lang = batch
        t2 = time.time()

        if model.module.lang_cond:
            context = b_lang * 5
        else:
            context = None

        ## Encode Start and End Frames
        bs = b_im.shape[0]
        b_im_r = b_im.reshape(bs*5, 3, 224, 224)
        alles, att = model(b_im_r, context)
        alle = alles.reshape(bs, 5, -1)
        e0 = alle[:, 0]
        eg = alle[:, 1]
        es0 = alle[:, 2]
        es1 = alle[:, 3]
        es2 = alle[:, 4]

        full_loss = 0
        langstuff, smoothstuff = None, None

        ## LP Loss
        l2loss = torch.linalg.norm(alles, ord=2, dim=-1).mean()
        if att is not None:
            l1loss = torch.linalg.norm(att, ord=1, dim=-1).mean()
            l0loss = torch.linalg.norm(att, ord=0, dim=-1).mean()
        else:
            l1loss = torch.linalg.norm(alles, ord=1, dim=-1).mean()
            l0loss = torch.linalg.norm(alles, ord=0, dim=-1).mean()
        metrics['l2loss'] = l2loss.item()
        metrics['l1loss'] = l1loss.item()
        metrics['l0loss'] = l0loss.item()
        full_loss += model.module.l2weight * l2loss
        if model.module.anneall1:
            currl1weight = min(1.0, (step / 1000000.0)) * model.module.l1weight
            full_loss += currl1weight * l1loss
        else:
            full_loss += model.module.l1weight * l1loss
 

        t3 = time.time()
        ## Language Predictive Loss
        if model.module.langweight > 0:
            num_neg = 3
            b_lang_shuf = copy.deepcopy(b_lang)

            sim_pos, _ = model.module.get_reward(e0, eg, b_lang)
            sim_negs = []
            sim_negs.append(model.module.get_reward(e0, es0, b_lang)[0])
            sim_negs.append(model.module.get_reward(e0, es1, b_lang)[0])
            sim_negs.append(model.module.get_reward(e0, es2, b_lang)[0])
            for _ in range(num_neg):
                random.shuffle(b_lang_shuf)
                sim_negs.append(model.module.get_reward(e0, eg, b_lang_shuf)[0])
            sim_negs = torch.stack(sim_negs, -1)
            sim_negs_exp = torch.exp(sim_negs)

            rewloss = -torch.log(epsilon + (torch.exp(sim_pos) / (epsilon + torch.exp(sim_pos) + sim_negs_exp.sum(-1)))).mean()
            lacc = (1.0 * (sim_negs.max(-1)[0] < sim_pos)).mean()
            metrics['rewloss'] = rewloss.item()
            metrics['rewacc'] = lacc.item()

            full_loss += model.module.langweight * rewloss
            langstuff = (e0, eg, sim_pos)

        t4 = time.time()
        ## Cross Video Contrative Loss
        if model.module.cpcweight > 0:
            sim_0_2 = model.module.sim(es2, es0) 
            p_s = []
            n_s = []
            for k in range(0, bs, model.module.num_same):
                subbatch = es0[k:(k+model.module.num_same)]
                pos_s = sim_0_2[k]
                p_s.append(pos_s)
                neg_s = []
                for sb in range(1, model.module.num_same):
                    neg_s.append(model.module.sim(es0[0].unsqueeze(0), es0[sb].unsqueeze(0)))
                neg_s = torch.cat(neg_s)
                n_s.append(neg_s)
            p_s = torch.stack(p_s)     
            n_s = torch.stack(n_s)                
            cpcloss = -torch.log(epsilon + (torch.exp(p_s) / (epsilon + torch.exp(n_s).sum(-1) + torch.exp(p_s)))).mean()
            nmx, _ = torch.max(n_s, -1)
            cpcacc = (1.0 * (p_s > nmx)).mean()
            metrics['cpcloss'] = cpcloss.item()
            metrics['cpcacc'] = cpcacc.item()
            full_loss += model.module.cpcweight * cpcloss

        t5 = time.time()
        ## Within Video TCN Loss
        if model.module.tcnweight > 0:
            sim_0_2 = model.module.sim(es2, es0) 
            sim_1_2 = model.module.sim(es2, es1)
            sim_0_1 = model.module.sim(es1, es0)

            smoothloss1 = -torch.log(epsilon + (torch.exp(sim_1_2) / (epsilon + torch.exp(sim_0_2) + torch.exp(sim_1_2))))
            smoothloss2 = -torch.log(epsilon + (torch.exp(sim_0_1) / (epsilon + torch.exp(sim_0_1) + torch.exp(sim_0_2))))
            smoothloss = ((smoothloss1 + smoothloss2) / 2.0).mean()
            a_state = ((1.0 * (sim_0_2 < sim_1_2)) * (1.0 * (sim_0_1 > sim_0_2))).mean()
            metrics['tcnloss'] = smoothloss.item()
            metrics['aligned'] = a_state.item()
            full_loss += model.module.tcnweight * smoothloss
            smoothstuff = (es0, es1, es2, att)

        metrics['full_loss'] = full_loss.item()
        
        t6 = time.time()
        if not eval:
            model.module.encoder_opt.zero_grad()
            full_loss.backward()
            model.module.encoder_opt.step()

        if (step % self.eval_freq == 0):
            self.log_batch(b_im[:, 0], b_im[:, 1], b_im[:, 2], b_im[:, 3], b_im[:, 4], step, b_lang, eval, langstuff, smoothstuff)
        t7 = time.time()
        st = f"Load time {t1-t0}, Batch time {t2-t1}, Encode and LP tine {t3-t2}, Lang time {t4-t3}, Contrastive time {t5-t4}, TCN time {t6-t5}, Backprop time {t7-t6}"
        return metrics, st

    def update_simclr(self, model, batch, step, eval=False):
        t0 = time.time()
        metrics = dict()
        if eval:
            model.eval()
        else:
            model.train()

        t1 = time.time()
        ## Batch
        imv1, imv2 = batch
        t2 = time.time()

        ## Encode Start and End Frames
        bs = imv1.shape[0]
        ### Order such that 5 states from same video are in order
        ### That way, if split across gpus, still get distribution of time and video in each batch
        ## Put positive examples on different GPU when multibatch to prevent cheating
        bim = torch.stack([imv1, imv2], 0)
        bim = bim.reshape(2*bs, 3, 224, 224)
        contextlist = torch.tensor(range(0, bs*2))
        alles, _ = model(bim)
        alle = alles.reshape(2, bs, -1)
        ev1 = alle[0]
        ev2 = alle[1]

        full_loss = 0

        t3 = time.time()
        ## SimCLR loss
        temperature = 0.1
        pos = model.module.sim(ev1, ev2) / temperature
        neg = model.module.sim(ev1.repeat(bs, 1), ev1.repeat_interleave(bs, 0)) / temperature
        neg = neg.reshape(bs, bs)
        neg = neg.masked_select(~torch.eye(bs, dtype=bool).cuda()).reshape((bs, bs-1))

        neg2 = model.module.sim(ev1.repeat(bs, 1), ev2.repeat_interleave(bs, 0)) / temperature
        neg2 = neg2.reshape(bs, bs)
        neg2 = neg2.masked_select(~torch.eye(bs, dtype=bool).cuda()).reshape((bs, bs-1))

        contrastive_loss = -torch.log(epsilon + (torch.exp(pos) / 
                    (epsilon + torch.exp(neg).sum(-1) + torch.exp(neg2).sum(-1) + torch.exp(pos)))).mean()
        full_loss += contrastive_loss    
        acc = (1.0 * ((pos > neg2.max(-1)[0]) * (pos > neg.max(-1)[0]))).mean()
        metrics['contrastive_loss'] = contrastive_loss.item() 
        metrics['acc'] = acc.item() 
        
        t4 = time.time()
        metrics['full_loss'] = full_loss.item()
        
        t5 = time.time()
        if not eval:
            model.module.encoder_opt.zero_grad()
            full_loss.backward()
            model.module.encoder_opt.step()
            model.module.sched.step()

        t6 = time.time()
        st = f"Load time {t1-t0}, Batch time {t2-t1}, Encode time {t3-t2}, Loss time {t4-t3},  Backprop time {t6-t5}"

        return metrics, st

    ## Logging initial and final images and their language
    def log_data(self, b_im0, b_img, step, lang, eval):
        im_logging = []
        for i in range(8):
            im = torch.cat([b_im0[i], b_img[i]], 1)
            im_logging.append(im)
        ims_log = torch.cat(im_logging, -1) / 255.0
        self.work_dir = Path.cwd().joinpath('ims').joinpath(f'{eval}_{step}')
        self.work_dir.mkdir(parents=True, exist_ok=True)
        save_image(ims_log, self.work_dir.joinpath(f"im.png"))
        with open(self.work_dir.joinpath(f"lang.txt"), 'w') as f:
            for item in lang[:8]:
                f.write("%s\n" % item)

    ## Logging images used for TCN Loss
    def log_smooth_data(self, b_ims0, b_ims1, b_ims2, step, lang, eval):
        im_logging = []
        for i in range(8):
            im = torch.cat([b_ims0[i], b_ims1[i], b_ims2[i]], 1)
            im_logging.append(im)
        ims_log = torch.cat(im_logging, -1) / 255.0
        self.work_dir = Path.cwd().joinpath('ims').joinpath(f'{eval}_{step}')
        self.work_dir.mkdir(parents=True, exist_ok=True)
        save_image(ims_log, self.work_dir.joinpath(f"smoothim.png"))

    ## Logging language predictions
    def log_lang(self, tensors, step, eval):
        e0, eg, preds = tensors
        with open(self.work_dir.joinpath(f"preds.txt"), 'w') as f:
            for item in preds.squeeze():
                f.write("%s\n" % item)

    ### Plotting TCN triplets in embedding space and attention mask
    def log_smooth(self, tensors, step, eval):
        e0, e1, e2, attn = tensors
        bs = 10
        al = torch.cat([e0, e1, e2], 0)
        save_image(F.sigmoid(al / al.max(-1)[0].unsqueeze(-1)), self.work_dir.joinpath(f"embeddings.png"))
        u, s, v = torch.pca_lowrank(al, q=2, niter=100)
        u = u.cpu().detach().numpy()
        u_0 = u[0:bs]
        u_1 = u[bs:(2*bs)]
        u_2 = u[(2*bs):(3*bs)]
        for j in range(bs):
            plt.plot([u_0[j, 0], u_1[j, 0]], [u_0[j, 1], u_1[j, 1]], color="green", marker=".")
            plt.plot([u_1[j, 0], u_2[j, 0]], [u_1[j, 1], u_2[j, 1]], color="red", marker=".")
        plt.savefig(self.work_dir.joinpath(f"pca_smooth.png"))
        plt.close()

        if attn is not None:
            if len(attn.shape) == 2:
                save_image(attn, self.work_dir.joinpath(f"attn_smooth.png"))
            else:
                for j in range(bs):
                    save_image(attn[j].unsqueeze(0), self.work_dir.joinpath(f"attn_{j}_smooth.png"))

    def log_batch(self, b_im0, b_img, b_ims0, b_ims1, b_ims2, step, lang, eval, tensors=None, smoothtensors=None):
        ## Visualize Training Data
        self.log_data(b_im0, b_img, step, lang, eval)
        
        ## Visualize Language Data
        if tensors is not None:
            self.log_lang(tensors, step, eval)

        ## Visualize Smoothness Data
        if smoothtensors is not None:
            self.log_smooth(smoothtensors, step, eval)
            self.log_smooth_data(b_ims0, b_ims1, b_ims2, step, lang, eval)

