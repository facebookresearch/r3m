import logging
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from textwrap import wrap

from model import MLPBN
from robopen_dataset import (TRAINING_DATASET_DIR, 
                             GraspReleaseDataset, 
                             denormalize_y_cartesian)

sns.set()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

TEST_IMAGE_DIR = "/mnt/nfs2/giriman/data/policy/output/test_display/"
POLICY_OUTPUT_DIR = "/mnt/nfs2/giriman/data/policy/output/checkpoints"
LEARNING_RATE = 1e-4  

class HyperParameters():
    def __init__(self, epochs, mbs, img_aug_type) -> None:
        self.epochs = epochs
        self.mini_batch_size = mbs
        self.lr = LEARNING_RATE
        self.train_split = 0.95
        self.img_aug_type = img_aug_type

class GraspDataloader(torch.utils.data.Dataset):
    def __init__(self, grasp_release_dataset, img_aug_key):    
        self.grasp_release_dataset = grasp_release_dataset
        self.img_aug_key = img_aug_key

    def __len__(self):
        return self.grasp_release_dataset.dataset_length(self.img_aug_key)

    def __getitem__(self, i):
        return self.grasp_release_dataset.get_grasp_datapoint(i, self.img_aug_key)

class ReleaseDataloader(torch.utils.data.Dataset):
    def __init__(self, grasp_release_dataset, img_aug_key):    
        self.grasp_release_dataset = grasp_release_dataset
        self.img_aug_key = img_aug_key

    def __len__(self):
        return self.grasp_release_dataset.dataset_length(self.img_aug_key)

    def __getitem__(self, i):
        return self.grasp_release_dataset.get_release_datapoint(i, self.img_aug_key)

def param_str(model, hyperparameters):
    num_params = model.count_parameters()
    model_tag = model.tag()
    return (f"{model_tag}_is_{model.input_size}_os_{model.output_size}_nh_{model.num_hidden}"
            f"_pc_{num_params}_mbs_{hyperparameters.mini_batch_size}_lr_{LEARNING_RATE}"
            f"_epochs_{hyperparameters.epochs}_aug_{hyperparameters.img_aug_type}")    

def split_train_test(full_dataset, split=0.95):
    train_size = int(split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def saveModel(model, robopen_id, dir_name, file_name): 
    os.makedirs(POLICY_OUTPUT_DIR, exist_ok=True)
    rp_dir_name = os.path.join(POLICY_OUTPUT_DIR, robopen_id)
    os.makedirs(rp_dir_name, exist_ok=True)
    full_dir_name = os.path.join(rp_dir_name, dir_name)
    os.makedirs(full_dir_name, exist_ok=True)
    full_file_name = os.path.join(full_dir_name, file_name)    
    torch.save(model.state_dict(), full_file_name) 
    return full_file_name

def save_model_stats(stats, robopen_id, dir_name, file_name):
    os.makedirs(POLICY_OUTPUT_DIR, exist_ok=True)
    rp_dir_name = os.path.join(POLICY_OUTPUT_DIR, robopen_id)
    os.makedirs(rp_dir_name, exist_ok=True)
    full_dir_name = os.path.join(rp_dir_name, dir_name)
    os.makedirs(full_dir_name, exist_ok=True)
    full_file_name = os.path.join(full_dir_name, file_name)
    pickle.dump(stats, open(full_file_name, 'wb'))

def save_figure(fig, robopen_id, dir_name, file_name):
    os.makedirs(POLICY_OUTPUT_DIR, exist_ok=True)
    rp_dir_name = os.path.join(POLICY_OUTPUT_DIR, robopen_id)
    os.makedirs(rp_dir_name, exist_ok=True)
    full_dir_name = os.path.join(rp_dir_name, dir_name)
    os.makedirs(full_dir_name, exist_ok=True)
    full_file_name = os.path.join(full_dir_name, file_name)    
    fig.savefig(full_file_name, dpi=fig.dpi)

def plot_predictions(targets, predictions, robopen_id, dir_name, file_name):
    # Sampling params
    # GP_RANGE_UPPER = [0.7, 0.1, np.pi / 2]
    # GP_RANGE_LOWER = [0.4, -0.1, -np.pi / 2]    
    targets_df = pd.DataFrame(targets, columns = ['X','Y','Z'])
    predictions_df = pd.DataFrame(predictions, columns = ['X','Y','Z'])
    bins = 30
    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(12, 12)     
    sns.histplot(data=targets_df, x="X", kde=True, bins=bins, color="skyblue", 
        label="TARGET-X", ax=axs[0,0])
    sns.histplot(data=predictions_df, x="X", kde=True, bins=bins, color="red", 
        label="PREDICT-X", ax=axs[0,1])
    sns.histplot(data=targets_df, x="Y", kde=True, bins=bins, color="skyblue", 
        label="TARGET-Y", ax=axs[1,0])  
    sns.histplot(data=predictions_df, x="Y", kde=True, bins=bins, color="red", 
        label="PREDICT-Y", ax=axs[1,1]) 
    sns.histplot(data=targets_df, x="Z", kde=True, bins=bins, color="skyblue", 
        label="TARGET-Z", ax=axs[2,0]) 
    sns.histplot(data=predictions_df, x="Z", kde=True, bins=bins, color="red", 
        label="PREDICT-Z", ax=axs[2,1])  
    axs[0,0].legend(loc="upper right") 
    axs[0,1].legend(loc="upper right")    
    axs[1,0].legend(loc="upper right")   
    axs[1,1].legend(loc="upper right")
    axs[2,0].legend(loc="upper right")
    axs[2,1].legend(loc="upper right")   
    save_figure(fig, robopen_id, dir_name, file_name)
    plt.close(fig) 

def plot_loss_curves(train_stats, robopen_id, dir_name):
    epochs = train_stats["epochs"]
    title = f"{dir_name}"
    epochs = []
    train_loss = []
    test_loss = []
    epoch_stats = train_stats["epoch_stats"]
    for i in range(0,len(epoch_stats)):
        epoch = epoch_stats[i]["epoch"]
        epochs.append(epoch)
        train_loss.append(epoch_stats[i]["average_training_loss"])
        test_loss.append(epoch_stats[i]["average_test_loss"])
    fig, ax = plt.subplots(1)
    fig.set_size_inches(12, 12)    
    x = epochs
    ax.plot(x, train_loss, label="train")
    ax.plot(x, test_loss, label="test")
    ax.set_title("\n".join(wrap(title.upper())))
    ax.legend(loc="upper right")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")  
    mod_title = title.replace(" ", "_")   
    file_name = f"loss_stats_{mod_title.upper()}.png"
    save_figure(fig, robopen_id, dir_name, file_name)
    plt.close(fig)    

def inference(model, hyperparameters, dataset, device="cpu"):
    mbs = 1 # min(hyperparameters.mini_batch_size, len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset, 
        batch_size=mbs, 
        shuffle=True, num_workers=1, drop_last=True)  
    loss_function = nn.MSELoss()
    loss = 0
    all_targets = []
    all_targets_norm = []
    all_predictions_norm = []
    all_predictions = []
    model.eval()
    with torch.no_grad(): 
        for data in data_loader: 
            inputs, targets_norm, targets = data 
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            predicted_outputs = model(inputs) 
            loss += loss_function(predicted_outputs, targets)
            all_targets_norm = all_targets_norm + [x.tolist() for x in targets_norm]
            all_targets = all_targets + [x.tolist() for x in targets]
            all_predictions_norm = all_predictions_norm + \
                    [x.tolist() for x in predicted_outputs]
            all_predictions = all_predictions + \
                    [denormalize_y_cartesian(x).tolist() for x in predicted_outputs]
    return ((loss / (len(dataset) / mbs)), 
            all_targets,
            all_targets_norm, 
            all_predictions, 
            all_predictions_norm)

def test(test_dataset, model_file_name, prefix, hyperparameters): 
    test_stats = {}
    model = MLPBN() 
    path = model_file_name 
    model.load_state_dict(torch.load(path)) 
    average_test_loss, targets, targets_norm, \
        predictions, predictions_norm = inference(model, 
        hyperparameters, test_dataset)
    logging.info(f"{prefix} accuracy on test set of size "
                 f"{len(test_dataset)} is {average_test_loss}")
    test_stats["average_test_loss"] = average_test_loss
    test_stats["test_dataset_len"] = len(test_dataset)
    return test_stats, targets, targets_norm, predictions, predictions_norm

def train(model, device, train_dataset, 
         test_dataset, prefix, hyperparameters, 
         robopen_id, dir_name):
    train_stats = {}
    logging.info(f"{prefix} model will be training on {device} device") 
    logging.info(f"{param_str(model, hyperparameters)}") 
    train_stats["device"] = device
    model.to(device)      
    loss_function = nn.MSELoss()
    lr = hyperparameters.lr
    epochs = hyperparameters.epochs
    train_dataset_len = len(train_dataset) 
    logging.debug(f"{prefix} training dataset len {train_dataset_len}")
    mini_batch_size = hyperparameters.mini_batch_size
    train_stats["lr"] = lr
    train_stats["epochs"] = epochs
    train_stats["mini_batch_size"] = mini_batch_size
    train_stats["train_dataset_len"] = train_dataset_len
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
    train_start = time.time()
    epoch_stats = []
    avg_training_loss = float('inf')
    epoch = -1
    samples_seen = 0
    e_stats = {}
    e_stats["epoch"] = epoch
    e_stats["samples_seen"] = samples_seen
    e_stats["average_training_loss"] = avg_training_loss
    average_test_loss, _, _, _, _ = inference(model, hyperparameters, 
        test_dataset, device)
    e_stats["average_test_loss"] = average_test_loss.cpu()
    e_stats["test_dataset_len"] = len(test_dataset)   
    epoch_stats.append(e_stats)
    logging.info(f"{prefix} end of epoch {epoch}, samples trained {samples_seen}, "
                 f"average training loss {avg_training_loss} "
                 f"average test loss {average_test_loss}")
    for epoch in range(0, epochs+1):
        e_stats = {}
        logging.debug(f"Starting epoch {epoch}")
        train_loader = torch.utils.data.DataLoader(train_dataset, 
            batch_size=mini_batch_size, shuffle=True, num_workers=8, drop_last=True) 
        training_loss = 0.0
        samples_seen = 0.0
        batches_seen = 0
        all_targets = []
        all_targets_norm = []
        all_outputs_norm = []
        all_outputs = []        
        for i, data in enumerate(train_loader, 0):
            inputs, targets_norm, targets = data
            inputs, targets_norm = inputs.float().to(device), targets_norm.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets_norm)      
            loss.backward()
            optimizer.step()
            batches_seen += 1
            samples_seen += len(targets_norm)
            training_loss += loss.item()
            avg_training_loss = training_loss / batches_seen
            all_targets_norm = all_targets_norm + [x.tolist() for x in targets_norm]
            all_targets = all_targets + [x.tolist() for x in targets]
            all_outputs_norm = all_outputs_norm + [x.tolist() for x in outputs]
            all_outputs = all_outputs + [denormalize_y_cartesian(x).tolist() for x in outputs]
        e_stats["epoch"] = epoch
        e_stats["samples_seen"] = samples_seen
        e_stats["average_training_loss"] = round(avg_training_loss, 4)
        model.eval()
        average_test_loss, test_targets, test_targets_norm, \
            test_predictions, test_predictions_norm = \
                inference(model, hyperparameters, test_dataset, device)
        e_stats["average_test_loss"] = average_test_loss.cpu()
        e_stats["test_dataset_len"] = len(test_dataset)        
        epoch_stats.append(e_stats)
        model.train()
        if (epoch == 0) or (epoch != 0 and (epoch & (epoch-1) == 0)):
            logging.info(f"{prefix} end of epoch {epoch}, samples trained {(epoch + 1)*samples_seen}" 
                         f"average training loss {avg_training_loss} average test loss {average_test_loss}")
            plot_predictions(test_targets, test_predictions, robopen_id, dir_name, 
                f"target_vs_predictions_epoch_{epoch}.png")
            plot_predictions(test_targets_norm, test_predictions_norm, robopen_id, dir_name, 
                f"target_vs_predictions_norm_epoch_{epoch}.png")
            plot_predictions(all_targets, all_outputs, robopen_id, dir_name, 
                f"train_vs_outputs_epoch_{epoch}.png")
            plot_predictions(all_targets_norm, all_outputs_norm, robopen_id, dir_name, 
                f"train_vs_outputs_norm_epoch_{epoch}.png")

    train_end = time.time()
    train_time_seconds = train_end - train_start
    train_stats["epoch_stats"] = epoch_stats
    train_stats["train_time_seconds"] = train_time_seconds
    train_stats["input_size"] = model.input_size
    train_stats["output_size"] = model.output_size
    train_stats["num_parameters"] = model.count_parameters()
    logging.info(f"{prefix} - training on {train_dataset_len} samples for {epochs} epochs ",
                 f"on {device} to average training loss {avg_training_loss} "
                 f"took {train_time_seconds} seconds") 
    plot_loss_curves(train_stats, robopen_id, dir_name)
    return train_stats   

def train_test(data_loader, robopen_id, prefix, hyperparameters):
    stats = {}
    train_dataset, test_dataset = split_train_test(data_loader, split=hyperparameters.train_split)
    model = MLPBN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    p_str = param_str(model, hyperparameters)
    dir_name = f"{prefix}_{p_str}"
    train_stats = train(model, device, train_dataset, test_dataset, 
        prefix, hyperparameters, robopen_id, dir_name)
    model_file_name = f"model_{prefix}_{p_str}.pt"
    full_model_file_name = saveModel(model, robopen_id, dir_name, model_file_name)
    test_stats, test_targets, test_targets_norm, \
        test_predictions, test_predictions_norm = \
            test(test_dataset, full_model_file_name, prefix, hyperparameters)
    stats["train_stats"] = train_stats
    stats["test_stats"] = test_stats
    stats["prefix"] = prefix
    stats["robopen_id"] = robopen_id
    stats_file_name = f"stats_{prefix}_{p_str}.pickle"
    save_model_stats(stats, robopen_id, dir_name, stats_file_name)
    plot_predictions(test_targets, test_predictions, robopen_id, dir_name, 
        "target_vs_predictions_final.png")
    plot_predictions(test_targets_norm, test_predictions_norm, robopen_id, dir_name, 
        "target_vs_predictions_norm_final.png")

def train_datasets():
    tags = ["robopen05", "robopen04", "robopen03", "robopen02"]
    epochs = 2048
    mbs_vals = [512, 256, 128]
    mbs_vals.reverse()
    for tag in tags:
        dataset_file_name = f"{tag}_clp_dataset.pickle"
        os.makedirs(TRAINING_DATASET_DIR, exist_ok=True)        
        full_dataset_file_name = os.path.join(TRAINING_DATASET_DIR, dataset_file_name)
        grasp_release_dataset = GraspReleaseDataset()
        grasp_release_dataset.load_dataset(full_dataset_file_name)
        img_augmentation_keys = grasp_release_dataset.img_augmentation_keys + ["all"]
        for aug_key in img_augmentation_keys:
            for mbs in mbs_vals:
                torch.manual_seed(42)
                hyperparameters = HyperParameters(epochs, mbs, aug_key)
                grasp_data_loader = GraspDataloader(grasp_release_dataset, aug_key)
                train_test(grasp_data_loader, tag, f"{tag}_grasp", hyperparameters)
                torch.manual_seed(42)
                release_data_loader = ReleaseDataloader(grasp_release_dataset, aug_key)
                train_test(release_data_loader, tag, f"{tag}_release", hyperparameters)

if __name__ == '__main__':
    train_datasets()
