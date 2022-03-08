from robolang_rep.models_r3m import R3M

import os 
from os.path import expanduser
import omegaconf
import hydra
import gdown
import torch

def load_r3m(modelid):
    home = os.path.join(expanduser("~"), ".r3m")
    if modelid == "resnet50":
        foldername = "r3m_50"
        modelurl = 'https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA'
        configurl = 'https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8'
    elif modelid == "resnet34":
        foldername = "r3m_34"
        modelurl = 'https://drive.google.com/uc?id=15bXD3QRhspIRacOKyWPw5y2HpoWUCEnE'
        configurl = 'https://drive.google.com/uc?id=1RY0NS-Tl4G7M1Ik_lOym0b5VIBxX9dqW'
    elif modelid == "resnet18":
        foldername = "r3m_18"
        modelurl = 'https://drive.google.com/uc?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-'
        configurl = 'https://drive.google.com/uc?id=1nitbHQ-GRorxc7vMUiEHjHWP5N11Jvc6'
    else:
        raise NameError('Invalid Model ID')

    if not os.path.exists(os.path.join(home, foldername)):
        os.makedirs(os.path.join(home, foldername))
    modelpath = os.path.join(home, foldername, "model.pt")
    configpath = os.path.join(home, foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    rep = hydra.utils.instantiate(modelcfg.agent)
    rep = torch.nn.DataParallel(rep)
    rep.load_state_dict(torch.load(modelpath)['r3m'])
    return rep

def load_r3m_reproduce(modelid):
    home = os.path.join(expanduser("~"), ".r3m")
    if modelid == "r3m":
        foldername = "original_r3m"
        modelurl = 'https://drive.google.com/uc?id=1jLb1yldIMfAcGVwYojSQmMpmRM7vqjp9'
        configurl = 'https://drive.google.com/uc?id=1cu-Pb33qcfAieRIUptNlG1AQIMZlAI-q'
    elif modelid == "r3m_noaug":
        foldername = "original_r3m_noaug"
        modelurl = 'https://drive.google.com/uc?id=1k_ZlVtvlktoYLtBcfD0aVFnrZcyCNS9D'
        configurl = 'https://drive.google.com/uc?id=1hPmJwDiWPkd6GGez6ywSC7UOTIX7NgeS'
    elif modelif == "r3m_nol1":
        foldername = "original_r3m_nol1"
        modelurl = 'https://drive.google.com/uc?id=1LpW3aBMdjoXsjYlkaDnvwx7q22myM_nB'
        configurl = 'https://drive.google.com/uc?id=1rZUBrYJZvlF1ReFwRidZsH7-xe7csvab'
    elif modelif == "r3m_nolang":
        foldername = "original_r3m_nolang"
        modelurl = 'https://drive.google.com/uc?id=1FXcniRei2JDaGMJJ_KlVxHaLy0Fs_caV'
        configurl = 'https://drive.google.com/uc?id=192G4UkcNJO4EKN46ECujMcH0AQVhnyQe'
    else:
        raise NameError('Invalid Model ID')

    if not os.path.exists(os.path.join(home, foldername)):
        os.makedirs(os.path.join(home, foldername))
    modelpath = os.path.join(home, foldername, "model.pt")
    configpath = os.path.join(home, foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    rep = hydra.utils.instantiate(modelcfg.agent)
    rep = torch.nn.DataParallel(rep)
    rep.load_state_dict(torch.load(modelpath)['r3m'])
    return rep
    