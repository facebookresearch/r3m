import omegaconf
import hydra
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

from robolang_rep import load_r3m

device = "cuda"

r3m = load_r3m("resnet50") # resnet18, resnet34
r3m.eval()
r3m.to(device)

## DEFINE PREPROCESSING
transforms = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()]) # ToTensor() divides by 255

## ENCODE IMAGE
image = np.random.randint(0, 255, (500, 500, 3))
preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(-1, 3, 224, 224)
preprocessed_image.to(device) 
embedding = r3m(preprocessed_image * 255.0)[0] ## R3M expects image input to be [0-255]
print(embedding.shape) # [1, 2048]