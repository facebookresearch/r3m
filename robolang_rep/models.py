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
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Identity
import torchvision
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel, AutoConfig
# import utils
from robolang_rep import utils
import random
from pathlib import Path
from torchvision.utils import save_image
import time
import copy
import matplotlib.pyplot as plt
import torchvision.transforms as T
import sklearn.metrics
import wandb

epsilon = 1e-8
def do_nothing(x): return x

class ResnetEncoder(nn.Module):
    def __init__(self, obs_shape, finetune, pretrained, reshape=0, 
                num_ims=3, size=18, proprio_shape=0, lang_cond=False, 
                lang_enc=None, attntype="modulate", lsize=32):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_ims = num_ims
        self.obs_shape = obs_shape
        self.lang_cond = lang_cond
        self.lsize = lsize
        self.attntype = attntype
        self.proprio_shape = proprio_shape
        self.finetune = finetune

        if size in [18, 34]:
            self.output_dim = 512
        elif size == 50:
            self.output_dim = 2048
        self.repr_dim = self.output_dim * num_ims + proprio_shape
        
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

        ### IF REPRESENTATION IS LANGUAGE CONDITIONED
        if self.lang_cond:
            self.lang_enc = lang_enc
            if (self.attntype == "modulate") or (self.attntype == "modulatesigm"):
                self.lang_enc_2 = nn.Linear(self.lang_enc.lang_size, self.output_dim)
            elif (self.attntype == "sigmoid") or (self.attntype == "fc"):
                self.attn = nn.Sequential(nn.Linear(self.lang_enc.lang_size + self.output_dim, self.output_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.output_dim, self.output_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.output_dim, self.output_dim))
            elif (self.attntype == "ef"):
                self.convnet.conv1 = nn.Conv2d(3 + self.lsize, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.lang_enc_2 = nn.Linear(self.lang_enc.lang_size, self.lsize)
            elif (self.attntype == "mh"):
                self.heads = 1 ## One head scaled dot product attention
                self.attn = torch.nn.MultiheadAttention(self.heads, self.heads, batch_first=True)
                self.lang_enc_2 = nn.Linear(self.lang_enc.lang_size, self.output_dim)
            self.sigm = Sigmoid()

        if not pretrained:
            self.apply(utils.weight_init)

    def encode(self, image, sentences=None):
        ## If early fusion
        if (self.lang_cond) and (self.attntype == "ef"):
            ## Encode sentences to lower dim
            le = self.lang_enc_2(self.lang_enc(sentences))
            le = le.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, image.shape[2], image.shape[3]) 
            ## Stack channelwise and encode
            image_c = torch.cat([image, le], 1)
            e = self.convnet(image_c).squeeze()
            return e, (None, None, None)

        ### Encode with Resnet
        e = self.convnet(image).squeeze(-1).squeeze(-1)
        a = None
        lpred, ltrue = None, None
        ## If language conditioned
        if self.lang_cond:
            ltrue = self.lang_enc(sentences)
            ## Modulate based on just language
            if self.attntype == "modulate":
                e = e * self.lang_enc_2(ltrue)
            ## Sigmoid mask just based on language
            elif self.attntype == "modulatesigm":
                a = self.sigm(self.lang_enc_2(ltrue))
                e = e * a
            ## Sigmoid mask based on image and language
            elif self.attntype == "sigmoid":
                a = self.sigm(self.attn(torch.cat([ltrue, e], -1)))
                e = e * a
            ## Fully connected language conditioning
            elif self.attntype == "fc":
                e = self.attn(torch.cat([ltrue, e], -1))
            ## Scaled dot product attention
            elif self.attntype == "mh":
                l_t = self.lang_enc_2(self.lang_enc(sentences))
                e_res = e.unsqueeze(-1)
                l_res = l_t.unsqueeze(-1)
                e, a = self.attn(l_res, e_res, e_res)
                e = e.mean(-1)
        return e.squeeze(), (a, lpred, ltrue)


    def forward(self, obs, sentences=None, contextlist=None):
        ## For multi-gpu training
        if (sentences is not None) and (contextlist is not None):
            ls = contextlist.squeeze().cpu().detach().numpy().astype(np.uint8)
            sentences = [sentences[i] for i in ls]

        ## If proprioceptive data is stacked at front of obs
        if self.proprio_shape > 0:
            proprio = obs[:, -self.proprio_shape:]
            obs = obs[:, :-self.proprio_shape].reshape([obs.shape[0]] + list(self.obs_shape))

        a = None
        h = []
        ## Encode and stack each image
        obs = obs.float() /  255.0
        for i in range(0, self.num_ims*3, 3):
            obs_p = self.preprocess(obs[:, i:(i+3)])
            if self.finetune:
                e, a = self.encode(obs_p, sentences)
            else:
                with torch.no_grad():
                    e, a = self.encode(obs_p, sentences)
            h.append(e)
        h = torch.cat(h, -1)
        h = h.view(h.shape[0], -1)
        ## Add back proprioception if there
        if self.proprio_shape > 0:
            h = torch.cat([h, proprio], -1)
        return (h, a)

class Representation:
    def __init__(self, device, lr, feature_dim,
                 hidden_dim, use_tb, finetune=1, pretrained=0, size=34, l2weight=1.0, 
                 langweight=1.0, tcnweight=0.0, structured=False, lang_cond=False, 
                 gt=False, l2dist=True, attntype="modulate", finetunelang=0, lsize=32, 
                 distributed=False, cpcweight = 0.0, num_same=1, langtype="reconstruct"):

        self.device = device
        self.use_tb = use_tb
        self.l2weight=l2weight
        self.lang_cond = lang_cond
        self.tcnweight = tcnweight
        self.cpcweight = cpcweight
        self.num_same = num_same
        self.distributed = distributed ## Multi-GPU Training
        self.finetunelang = finetunelang
        self.gt = gt
        self.l2dist = l2dist
        self.attntype = attntype
        self.langtype = langtype
        self.cs = torch.nn.CosineSimilarity(1)
        self.langweight = langweight
        self.lang_enc = LangEncoder(self.finetunelang, 0)
        self.encoder = ResnetEncoder((3, 224, 224), finetune, pretrained, reshape=0, 
                            num_ims=1, size=size, lang_cond=lang_cond, 
                            lang_enc=self.lang_enc, attntype=self.attntype, lsize=lsize).to(device)
        
        ## Types of language loss
        if self.langtype == "reconstruct":
            self.rew_model = RewardModel(self.encoder.repr_dim, hidden_dim, self.lang_enc.lang_size).to(device)
        elif self.langtype == "lorel":
            self.rew_model = RewardModel(self.encoder.repr_dim, hidden_dim, self.lang_enc.lang_size, disc=True).to(device)
        elif self.langtype == "contrastive":
            self.rew_model = LangPredictor(self.encoder.repr_dim, hidden_dim, self.lang_enc.lang_size).to(device)

        self.bce = nn.BCELoss(reduce=False)
        self.repr_dim = self.encoder.repr_dim

        if self.distributed:
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.lang_enc = torch.nn.DataParallel(self.lang_enc)
            self.rew_model = torch.nn.DataParallel(self.rew_model)

        if self.finetunelang:
            self.encoder_opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.lang_enc.parameters()) + list(self.rew_model.parameters()), lr=lr)
        else:
            self.encoder_opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.rew_model.parameters()), lr=lr)

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
                save_image(attn * 255.0, self.work_dir.joinpath(f"attn_smooth.png"))
            else:
                for j in range(bs):
                    save_image(attn[j].unsqueeze(0) * 255.0, self.work_dir.joinpath(f"attn_{j}_smooth.png"))

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

    ## Function for getting reward given language and initial/final states
    def get_reward(self, bim0, bims, lang):
        self.encoder.eval()
        self.rew_model.eval()

        if self.lang_cond:
            context = lang * 2
        else:
            context = None

        bs = bim0.shape[0]
        bim = torch.stack([bim0, bims], 0)
        bim = bim.reshape(2*bs, 3, 224, 224)
        alle, out = self.encoder(bim, context)
        att, _, ltrue = out
        alle = alle.reshape(2, bs, -1)
        e0 = alle[0]
        es = alle[1]

        if ltrue is None:
            ltrue = self.lang_enc(lang)


        if self.langtype == "reconstruct":
            preds = self.rew_model(e0, es)
            rew = -F.mse_loss(ltrue[:bs], preds, reduce=False).mean(-1)
        elif self.langtype == "lorel":
            rew = self.rew_model(e0, es, ltrue)
        elif self.langtype == "contrastive":
            goals = self.rew_model(e0, ltrue)
            rew = self.sim(goals, es)

        return rew, att

    def sim(self, tensor1, tensor2):
        if self.l2dist:
            d = - torch.linalg.norm(tensor1 - tensor2, dim = -1) #torch.sqrt(((es1 - es0)**2).mean(-1))
        else:
            d = self.cs(tensor1, tensor2)
        return d

    def update(self, batch, step, eval=False):
        t0 = time.time()
        metrics = dict()
        if eval:
            self.encoder.eval()
            self.rew_model.eval()
            self.lang_enc.eval()
        else:
            self.encoder.train()
            self.rew_model.train()
            if self.finetunelang:
                self.lang_enc.train()

        t1 = time.time()
        ## Batch
        b_im0, b_img, b_s0, b_s1, b_s2, b_lang = batch
        t2 = time.time()

        if self.lang_cond:
            context = b_lang * 5
        else:
            context = None

        ## Encode Start and End Frames
        bs = b_im0.shape[0]
        bim = torch.stack([b_im0, b_img, b_s0, b_s1, b_s2], 0)
        bim = bim.reshape(5*bs, 3, 224, 224)
        contextlist = torch.tensor(range(0, bs*5))
        alle, out = self.encoder(bim, context, contextlist)
        att, _, ltrue = out
        alle = alle.reshape(5, bs, -1)
        e0 = alle[0]
        eg = alle[1]
        es0 = alle[2]
        es1 = alle[3]
        es2 = alle[4]

        full_loss = 0
        langstuff, smoothstuff = None, None

        ## L1 Loss
        l1loss = (torch.linalg.norm(e0, ord=1) + torch.linalg.norm(eg, ord=1)) / 2.0
        metrics['l1loss'] = l1loss.item()
        full_loss += self.l2weight * l1loss
 

        t3 = time.time()
        ## Language Predictive Loss
        if self.langweight > 0:
            assert(not self.gt)

            nltrue = None
            if ltrue is None:
                ltrue = self.lang_enc(b_lang)

            if self.langtype == "contrastive":
                b_lang_shuf = copy.deepcopy(b_lang)
                random.shuffle(b_lang_shuf)
                if nltrue is None:
                    nltrue = self.lang_enc(b_lang_shuf)

                preds = self.rew_model(e0, ltrue[:bs])
                preds_n = self.rew_model(e0, nltrue[:bs])

                sim_0 = self.sim(preds, e0) 
                sim_n = self.sim(preds, preds_n)
                sim_g = self.sim(preds, eg)

                rewloss = -torch.log(epsilon + (torch.exp(sim_g) / (epsilon + torch.exp(sim_0) + torch.exp(sim_n) + torch.exp(sim_g)))).mean()
                lacc = ((1.0 * (sim_0 < sim_g)) * (1.0 * (sim_n < sim_g))).mean()
                metrics['rewloss'] = rewloss.item()
                metrics['rewacc'] = lacc.item()
            elif self.langtype == "lorel":
                b_lang_shuf = copy.deepcopy(b_lang)
                random.shuffle(b_lang_shuf)
                context_shuf = b_lang_shuf * 2
                contextlist_shuf = torch.tensor(range(0, bs*2))
                bimneg = torch.cat([b_im0, b_img], 0)
                alle_neg, nout = self.encoder(bimneg, context_shuf, contextlist_shuf)
                _, _, nltrue = nout
                if nltrue is None:
                    nltrue = self.lang_enc(b_lang_shuf)
                alle_neg = alle_neg.reshape(2, bs, -1)
                en0 = alle_neg[0]
                eng = alle_neg[1]
                

                pos = self.rew_model(e0, eg, ltrue[:bs])
                neg = self.rew_model(en0, eng, nltrue[:bs])
                neg2 = self.rew_model(eg, e0, ltrue[:bs])

                labels_pos = torch.ones(bs, 1).cuda()
                labels_neg = torch.zeros(bs*2, 1).cuda()
                preds = torch.cat([pos, neg, neg2], 0)
                labels = torch.cat([labels_pos, labels_neg], 0)
                rewloss = self.bce(preds, labels).mean()
                
                rewacc = (((1 * (preds > 0.5)) == labels) * 1.0).mean().cpu().detach().numpy()
                rewprec, rewrec, _, _ = sklearn.metrics.precision_recall_fscore_support(labels.cpu().detach().numpy(), 
                                                (1 * (preds > 0.5)).cpu().detach().numpy(), zero_division=0)
                metrics['rewloss'] = rewloss.item()
                metrics['rewacc'] = rewacc.item()
                metrics['rewprec'] = rewprec.mean().item()
                metrics['rewrec'] = rewrec.mean().item()
            elif self.langtype == "reconstruct":
                preds = self.rew_model(e0, eg)
                rewloss = F.mse_loss(ltrue[:bs], preds)
                metrics['rewloss'] = rewloss.item()

            full_loss += self.langweight * rewloss
            langstuff = (e0, eg, preds)

        ## Cross Video Contrative Loss
        if self.cpcweight > 0:
            sim_0_2 = self.sim(es2, es0) 

            p_s = []
            n_s = []
            for k in range(0, bs, self.num_same):
                subbatch = es0[k:(k+self.num_same)]
                pos_s = sim_0_2[k]
                p_s.append(pos_s)
                neg_s = []
                for sb in range(1, self.num_same):
                    neg_s.append(self.sim(es0[0].unsqueeze(0), es0[sb].unsqueeze(0)))
                neg_s = torch.cat(neg_s)
                n_s.append(neg_s)
            p_s = torch.stack(p_s)     
            n_s = torch.stack(n_s)                
            cpcloss = -torch.log(epsilon + (torch.exp(p_s) / (epsilon + torch.exp(n_s).sum(-1) + torch.exp(p_s)))).mean()
            nmx, _ = torch.max(n_s, -1)
            cpcacc = (1.0 * (p_s > nmx)).mean()
            metrics['cpcloss'] = cpcloss.item()
            metrics['cpcacc'] = cpcacc.item()
            full_loss += self.cpcweight * cpcloss

        ## Within Video TCN Loss
        t4 = time.time()
        if self.tcnweight > 0:
            assert(not self.gt)
            sim_0_2 = self.sim(es2, es0) 
            sim_1_2 = self.sim(es2, es1)
            sim_0_1 = self.sim(es1, es0)

            smoothloss1 = -torch.log(epsilon + (torch.exp(sim_1_2) / (epsilon + torch.exp(sim_0_2) + torch.exp(sim_1_2))))
            smoothloss2 = -torch.log(epsilon + (torch.exp(sim_0_1) / (epsilon + torch.exp(sim_0_1) + torch.exp(sim_0_2))))
            smoothloss = ((smoothloss1 + smoothloss2) / 2.0).mean()
            a_state = ((1.0 * (sim_0_2 < sim_1_2)) * (1.0 * (sim_0_1 > sim_0_2))).mean()
            metrics['tcnloss'] = smoothloss.item()
            metrics['aligned'] = a_state.item()
            full_loss += self.tcnweight * smoothloss
            smoothstuff = (es0, es1, es2, att)

        metrics['full_loss'] = full_loss.item()
        
        t5 = time.time()
        if not eval:
            self.encoder_opt.zero_grad()
            full_loss.backward()
            self.encoder_opt.step()

        if (step % self.eval_freq == 0):
            self.log_batch(b_im0, b_img, b_s0, b_s1, b_s2, step, b_lang, eval, langstuff, smoothstuff)
        t6 = time.time()
        # print(f"Aug time {t2-t1}, Encode A tine {t3-t2},  \
        #         Lang time {t4-t3}, Smooth time {t5-t4}, Backprop time {t6-t5}")

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
    def __init__(self, feature_dim, hidden_dim, lang_dim):
        super().__init__()
        self.pred = nn.Sequential(nn.Linear(feature_dim+lang_dim, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, feature_dim))
        
    def forward(self, emb, lang_emb):
        embin = torch.cat([emb, lang_emb], -1)
        emb_pred = self.pred(embin)
        return emb_pred

class RewardModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, lang_dim, disc=False):
        super().__init__()
        self.disc = disc
        if self.disc:
            self.pred = nn.Sequential(nn.Linear(feature_dim * 2 + lang_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, 1))
        else:
            self.pred = nn.Sequential(nn.Linear(feature_dim * 2, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, lang_dim))
        self.sigm = Sigmoid()

        
    def forward(self, e0, eg, l=None):
        if self.disc:
            embin = torch.cat([e0, eg, l], -1)
        else:
            embin = torch.cat([e0, eg], -1)
        out = self.pred(embin)
        if self.disc:
            return self.sigm(out)
        else:
            return out