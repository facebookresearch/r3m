import omegaconf
import hydra
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

load_path = "/checkpoint/surajn/drqoutput/train_representation/2022-01-24_11-10-28/10_agent.finetunelang=0,agent.l1weight=1e-05,agent.langtype=lorel,agent.langweight=1.0,agent.size=50,batch_size=16,dataset=ego4d,doaug=rctraj,experiment=rep_0124/snapshot_1000000.pt"

device = "cuda"

## READ CONFIG AND INSTANTIATE MODEL
parent_dir = "/".join(load_path.split("/")[:-1])
hydra_path = parent_dir + "/.hydra/config.yaml"
modelcfg = omegaconf.OmegaConf.load(hydra_path)
rep = hydra.utils.instantiate(modelcfg.agent)
rep = torch.nn.DataParallel(rep)

## LOAD WEIGHTS
rep.load_state_dict(torch.load(load_path)['r3m'])
rep.eval()
rep.to(device)

## DEFINE PREPROCESSING
transforms = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()]) # ToTensor() divides by 255

## ENCODE IMAGE
image = np.random.randint(0, 255, (500, 500, 3))
preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(-1, 3, 224, 224)
preprocessed_image.to(device) 
embedding = rep(preprocessed_image * 255.0)[0] ## R3M expects image input to be [0-255]
print(embedding.shape) # [1, 2048]