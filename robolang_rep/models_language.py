# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from transformers import AutoTokenizer, AutoModel, AutoConfig

epsilon = 1e-8

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

class LanguageReward(nn.Module):
    def __init__(self, ltype, im_dim, hidden_dim, lang_dim, simfunc=None):
        super().__init__()
        self.ltype = ltype
        self.sim = simfunc
        self.sigm = Sigmoid()
        if self.ltype == "contrastive":
            self.pred = nn.Sequential(nn.Linear(im_dim+lang_dim, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, im_dim))
        elif self.ltype == "lorel":
            self.pred = nn.Sequential(nn.Linear(im_dim * 2 + lang_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, 1))
        elif self.ltype == "reconstruct":
            self.pred = nn.Sequential(nn.Linear(im_dim * 2, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, lang_dim))
        
    def forward(self, e0, eg, le):
        info = {}
        if self.ltype == "contrastive":
            target = self.pred(torch.cat([e0, le], -1))
            info["target"] = target
            return self.sim(target, eg), info
        elif self.ltype == "lorel":
            return self.sigm(self.pred(torch.cat([e0, eg, le], -1))), info
        elif self.ltype == "reconstruct":
            lpred =  self.pred(torch.cat([e0, eg], -1))
            info["lpred"] = lpred
            return(self.sim(lpred, le))

class LanguageAttention(nn.Module):
    def __init__(self, attntype, im_dim, lang_dim, simfunc=None):
        super().__init__()
        self.attntype = attntype
        if (self.attntype == "modulate") or (self.attntype == "modulatesigm"):
            self.lang_enc_2 = nn.Linear(lang_dim, im_dim)
        elif (self.attntype == "sigmoid") or (self.attntype == "fc"):
            self.attn = nn.Sequential(nn.Linear(lang_dim + im_dim, im_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(im_dim, im_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(im_dim, im_dim))
        elif (self.attntype == "mh"):
            self.heads = 1 ## One head scaled dot product attention
            self.attn = torch.nn.MultiheadAttention(self.heads, self.heads, batch_first=True)
            self.lang_enc_2 = nn.Linear(lang_dim, im_dim)
        self.sigm = Sigmoid()
        
    def forward(self, e, ltrue):
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
        return e, a