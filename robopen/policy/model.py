import numpy as np
import torch
import torchvision.transforms as T
from r3m import load_r3m
from torch import nn


def initialize_r3m():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"   
    r3m = load_r3m("resnet50") # resnet18, resnet34
    r3m.eval()
    r3m.to(device)
    return r3m, device

def generate_embedding(rgb_img, device, r3m):
    # ToTensor() divides by 255
    transforms = T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor()]) 
    preprocessed_image = transforms(rgb_img).reshape(-1, 3, 224, 224)
    preprocessed_image.to(device) 
    ## R3M expects image input to be [0-255]
    embedding = r3m(preprocessed_image * 255.0) 
    embedding = embedding.cpu().detach().numpy()
    e_tensor = None
    if torch.is_tensor(embedding):
        e_tensor = torch.flatten(embedding)
    elif isinstance(embedding, (np.ndarray, np.generic)):
        e_tensor = torch.flatten(torch.from_numpy(embedding))
    else:
        raise Exception("unknown embedding type")
    return e_tensor

def uniform_sample(lower, upper):
    return lower + (upper - lower) * torch.rand_like(lower)

def normalize(v, A, B, C, D):
    # v belongs to [A,B] and we want to map to [C,D]
    # y = (X-A)/(B-A) * (D-C) + C
    return (v - A) / (B - A) * (D - C) + C

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class MLP(nn.Module):
    def __init__(self, input_size=6144, output_size=3):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        layer0 = nn.Linear(input_size, 64)
        layer1 = nn.Linear(64, 32)
        layer2 = nn.Linear(32, output_size)
        self.layers = nn.Sequential(
            layer0,
            nn.ReLU(),
            layer1,
            nn.ReLU(),
            layer2
        )
        self.num_hidden = 1
        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def tag(self):
        return "no_bn"


class MLPBN(nn.Module):
    def __init__(self, input_size=6144, output_size=3):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        flatten0 = nn.Flatten()
        bn0 = nn.BatchNorm1d(input_size)
        layer0 = nn.Linear(input_size, 64)
        layer1 = nn.Linear(64, 32)
        layer2 = nn.Linear(32, output_size)
        self.layers = nn.Sequential(
            flatten0,
            bn0,
            layer0,
            nn.ReLU(),
            layer1,
            nn.ReLU(),
            layer2
        )
        self.num_hidden = 1
        self.layers.apply(init_weights)        
        
    def forward(self, x):
        return self.layers(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def tag(self):
        return "bn"
