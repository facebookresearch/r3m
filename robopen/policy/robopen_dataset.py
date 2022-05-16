import io
import os
import pickle
import traceback

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from model import normalize, generate_embedding, initialize_r3m

# Sampling params - v1
RANDOM_GRASP_RANGE_UPPER_V1 = [0.7, -0.2, np.pi / 2]
RANDOM_GRASP_RANGE_LOWER_V1 = [0.4, -0.1, -np.pi / 2]
RANDOM_RELEASE_RANGE_UPPER_V1 = [0.7, 0.2, np.pi / 2]
RANDOM_RELEASE_RANGE_LOWER_V1 = [0.4, 0.1, -np.pi / 2]

# Sampling params - v2
RANDOM_GRASP_RANGE_UPPER_V2 = [0.75, -0.35, np.pi / 2]
RANDOM_GRASP_RANGE_LOWER_V2 = [0.35, -0.05, -np.pi / 2]
RANDOM_RELEASE_RANGE_UPPER_V2 = [0.75, 0.35, np.pi / 2]
RANDOM_RELEASE_RANGE_LOWER_V2 = [0.35, 0.05, -np.pi / 2]

TRAINING_DATASET_DIR = "/mnt/nfs2/giriman/data/policy/training_data/"

def identity(x):
    return x

def normalize_y_cartesian(y):
    # [C,D] in our case is [-1,1]
    dst_range = [-1, 1]
    y_arr = y.numpy()
    norm_y = [0, 0, 0]
    if y_arr[1] < 0:
        x_src_range = [RANDOM_GRASP_RANGE_LOWER_V2[0], RANDOM_GRASP_RANGE_UPPER_V2[0]]
        y_src_range = [RANDOM_GRASP_RANGE_LOWER_V2[1], RANDOM_GRASP_RANGE_UPPER_V2[1]]
        z_src_range = [RANDOM_GRASP_RANGE_LOWER_V2[2], RANDOM_GRASP_RANGE_UPPER_V2[2]]
        norm_y[0] = normalize(v=y_arr[0], A=x_src_range[0], B=x_src_range[1], C=dst_range[0], D=dst_range[1])
        norm_y[1] = normalize(v=y_arr[1], A=y_src_range[0], B=y_src_range[1], C=dst_range[0], D=dst_range[1])
        norm_y[2] = normalize(v=y_arr[2], A=z_src_range[0], B=z_src_range[1], C=dst_range[0], D=dst_range[1])
    else:
        x_src_range = [RANDOM_RELEASE_RANGE_LOWER_V2[0], RANDOM_RELEASE_RANGE_UPPER_V2[0]]
        y_src_range = [RANDOM_RELEASE_RANGE_LOWER_V2[1], RANDOM_RELEASE_RANGE_UPPER_V2[1]]
        z_src_range = [RANDOM_RELEASE_RANGE_LOWER_V2[2], RANDOM_RELEASE_RANGE_UPPER_V2[2]]
        norm_y[0] = normalize(v=y_arr[0], A=x_src_range[0], B=x_src_range[1], C=dst_range[0], D=dst_range[1])
        norm_y[1] = normalize(v=y_arr[1], A=y_src_range[0], B=y_src_range[1], C=dst_range[0], D=dst_range[1])
        norm_y[2] = normalize(v=y_arr[2], A=z_src_range[0], B=z_src_range[1], C=dst_range[0], D=dst_range[1])
    for i in range(0,len(norm_y)):
        if norm_y[i] < dst_range[0]:
            norm_y[i] = dst_range[0]
        if norm_y[i] > dst_range[1]:
            norm_y[i] = dst_range[1]
    return torch.flatten(torch.Tensor(norm_y))

def denormalize_y_cartesian(y):
    # [C,D] in our case is [-1,1]
    src_range = [-1, 1]
    x_dst_range = [0.75, 0.35]
    y_dst_range = [-0.35, 0.35]
    z_dst_range = [-np.pi / 2, np.pi / 2]
    norm_y = [0, 0, 0]
    norm_y[0] = normalize(v=y[0], A=src_range[0], B=src_range[1], C=x_dst_range[0], D=x_dst_range[1])
    norm_y[1] = normalize(v=y[1], A=src_range[0], B=src_range[1], C=y_dst_range[0], D=y_dst_range[1])
    norm_y[2] = normalize(v=y[2], A=src_range[0], B=src_range[1], C=z_dst_range[0], D=z_dst_range[1]) 
    return norm_y 

class GraspReleaseDataset():
    """
    Prepare the grasp release dataset for regression  
    """
    def __init__(self):
        self.num_cameras = 3
        self.num_joints = 7
        self.r3m, self.device = initialize_r3m()
        self.img_augmentation_keys = ["identity", "color_jitter", "gaussian_blur", "random_perspective"]
        self.img_augmentations = {
            "identity" : identity,
            "color_jitter" : T.ColorJitter(brightness=.5, hue=.3),
            "gaussian_blur" : T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            "random_perspective" : T.RandomPerspective(distortion_scale=0.6, p=1.0),
        }
        self.img_grasp_embeddings = {}
        self.img_release_embeddings = {}
        for aug_key in self.img_augmentation_keys:
            self.img_grasp_embeddings[aug_key] = []
            self.img_release_embeddings[aug_key] = []
        self.all_img_grasp_embeddings = []
        self.all_img_release_embeddings = []
        self.y_cartesian_grasp = []
        self.y_cartesian_release = []
        self.y_cartesian_grasp_normalized = []
        self.y_cartesian_release_normalized = []
        self.all_y_cartesian_grasp = []
        self.all_y_cartesian_release = []
        self.all_y_cartesian_grasp_normalized = []
        self.all_y_cartesian_release_normalized = []

    def validate_dataset(self):
        l1 = len(self.img_grasp_embeddings["identity"])
        l2 = len(self.img_release_embeddings["identity"])
        l3 = len(self.y_cartesian_grasp)
        l4 = len(self.y_cartesian_release)
        if l1 == l2 == l3 == l4:
            return True
        return False

    def save_dataset(self, file_name):
        data_set_dict = {}
        data_set_dict['grasps'] = {
            'embeddings' : self.img_grasp_embeddings,
            'cartesian_locations' : self.y_cartesian_grasp,
        }
        data_set_dict['releases'] = {
            'embeddings' : self.img_release_embeddings,
            'cartesian_locations' : self.y_cartesian_release,
        }
        pickle.dump(data_set_dict, open(file_name, 'wb'))

    def load_dataset(self, file_name):
        with open(file_name, "rb") as input_file:
            dataset = pickle.load(input_file)
            self.img_grasp_embeddings = dataset['grasps']['embeddings']
            self.y_cartesian_grasp = dataset['grasps']['cartesian_locations']     
            self.img_release_embeddings = dataset['releases']['embeddings']
            self.y_cartesian_release = dataset['releases']['cartesian_locations']   
            for y in self.y_cartesian_grasp:
                self.y_cartesian_grasp_normalized.append(normalize_y_cartesian(y))
            for y in self.y_cartesian_release:
                self.y_cartesian_release_normalized.append(normalize_y_cartesian(y))
            for _, v in self.img_grasp_embeddings.items():
                if len(v) > 0:
                    self.all_img_grasp_embeddings = self.all_img_grasp_embeddings + v
                    self.all_y_cartesian_grasp = self.all_y_cartesian_grasp + self.y_cartesian_grasp
                    self.all_y_cartesian_grasp_normalized = self.all_y_cartesian_grasp_normalized + \
                        self.y_cartesian_grasp_normalized
            for _, v in self.img_release_embeddings.items():
                if len(v) > 0 :
                    self.all_img_release_embeddings = self.all_img_release_embeddings + v
                    self.all_y_cartesian_release = self.all_y_cartesian_release + self.y_cartesian_release
                    self.all_y_cartesian_release_normalized = self.all_y_cartesian_release_normalized + \
                        self.y_cartesian_release_normalized
            print(f"all_img_grasp_embeddings {len(self.all_img_grasp_embeddings)}")
            print(f"all_y_cartesian_grasp {len(self.all_y_cartesian_grasp)}")

    def validate_camera_data_point(self, camera_features):
        retval = False
        if camera_features:
            features = [x for x in camera_features if x is not None]
            if len(features) == self.num_cameras:
                retval = True
        return retval

    def input_feature(self, embeddings):
        tensors = []
        for e in embeddings:
            e_tensor = None
            if torch.is_tensor(e):
                e_tensor = torch.flatten(e)
            elif isinstance(e, (np.ndarray, np.generic)):
                e_tensor = torch.flatten(torch.from_numpy(e))
            else:
                raise Exception("unknown embedding type")
            tensors.append(e_tensor)
        return torch.cat(tensors)

    def input_feature(self, embeddings):
        tensors = []
        for e in embeddings:
            e_tensor = None
            if torch.is_tensor(e):
                e_tensor = torch.flatten(e)
            elif isinstance(e, (np.ndarray, np.generic)):
                e_tensor = torch.flatten(torch.from_numpy(e))
            else:
                raise Exception("unknown embedding type")
            tensors.append(e_tensor)
        return torch.cat(tensors)

    def target_feature(self, f):
        if torch.is_tensor(f):
            return torch.flatten(f)
        elif isinstance(f, (np.ndarray, np.generic)):
            return torch.flatten(torch.from_numpy(f))
        elif isinstance(f, list):
            return torch.flatten(torch.Tensor(f))
        else:
            raise Exception("unknown target type" )           

    def process_compressed_images(self, imgs):
        retval = {}
        embeddings = {}
        try:
            for img in imgs:
                    img_png = Image.open(io.BytesIO(img)).convert("RGB")
                    for k, v in self.img_augmentations.items():
                        if k not in embeddings:
                            embeddings[k] = []
                        embeddings[k].append(generate_embedding(v(img_png), 
                            self.device, self.r3m))           
            for k, v in embeddings.items():
                retval[k] = self.input_feature(v)
        except Exception as e1:
            print(traceback.format_exc())    
            print(f"img augmentation failed - {str(e1)}") 
            retval = None
        return retval

    def add_image_data(self, file_name):
        try:  
            with open(file_name, "rb") as input_file:
                trajectory_state = pickle.load(input_file)        
                grasp_position = None
                release_position = None
                for k, v in trajectory_state.items():
                    if "grasp_pose" in k:
                        grasp_position = self.target_feature(v)
                    elif "release_pose" in k:
                        release_position = self.target_feature(v)
                if grasp_position is not None and release_position is not None:
                    grasp_feature_embeddings = None
                    release_feature_embeddings = None
                    for k, v in trajectory_state.items():
                        state_dict = None
                        if "start_gripper_state" == k:
                            if isinstance(v, list):
                                state_dict = v[0]
                        elif "open_gripper_state" == k:
                            if isinstance(v, list):
                                state_dict = v[-1]                            
                            else:
                                # assume dict
                                state_dict = v
                        if state_dict and isinstance(state_dict, dict):
                            for obs_key, obs_val in state_dict.items():
                                if "compressed_color_images" == obs_key:
                                    if self.validate_camera_data_point(obs_val):
                                        embedding_tensors = \
                                            self.process_compressed_images(obs_val)
                                        if k == "start_gripper_state":
                                            grasp_feature_embeddings = embedding_tensors
                                        elif k == "open_gripper_state":
                                            release_feature_embeddings = embedding_tensors
                    if grasp_feature_embeddings is not None \
                        and release_feature_embeddings is not None:
                        for k, v in grasp_feature_embeddings.items():
                            self.img_grasp_embeddings[k].append(v)
                        for k, v in release_feature_embeddings.items():
                            self.img_release_embeddings[k].append(v)
                        self.y_cartesian_grasp.append(grasp_position)
                        self.y_cartesian_release.append(release_position)
                else:
                    print(f"missing grasp / release coords, ignoring corrupt pickle file - {file_name}")   
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(str(e))
            print(f"ignoring corrupt pickle file - {file_name}")  
        print(f"Done processing {file_name}")

    def add_embedding_data(self, file_name):
        try:  
            with open(file_name, "rb") as input_file:
                trajectory_state = pickle.load(input_file)
                grasp_position = None
                release_position = None
                for k, v in trajectory_state.items():
                    if "grasp_pose" in k:
                        grasp_position = self.target_feature(v)
                    elif "release_pose" in k:
                        release_position = self.target_feature(v)
                if grasp_position is not None and release_position is not None:
                    grasp_feature_embeddings = None
                    release_feature_embeddings = None
                    for k, v in trajectory_state.items():
                        embedding_dict = None
                        if "start_gripper_state" == k:
                            if isinstance(v, list):
                                embedding_dict = v[0]
                        elif "open_gripper_state" == k:
                            if isinstance(v, list):
                                embedding_dict = v[-1]                            
                            else:
                                # assume dict
                                embedding_dict = v
                        if embedding_dict and isinstance(embedding_dict, dict):
                            for obs_key, obs_val in embedding_dict.items():
                                if "color_image_embeddings" == obs_key:
                                    if self.validate_camera_data_point(obs_val):
                                        embedding_tensor = self.input_feature(obs_val)
                                        if k == "start_gripper_state":
                                            grasp_feature_embeddings = \
                                                {"identity": embedding_tensor}
                                        elif k == "open_gripper_state":
                                            release_feature_embeddings = \
                                                {"identity": embedding_tensor}
                    if grasp_feature_embeddings is not None \
                        and release_feature_embeddings is not None:
                        for k, v in grasp_feature_embeddings:
                            self.img_grasp_embeddings[k].append(v)
                        for k, v in release_feature_embeddings:
                            self.img_release_embeddings[k].append(v)
                        self.y_cartesian_grasp.append(grasp_position)
                        self.y_cartesian_release.append(release_position)
                else:
                    print(f"missing grasp / release coords, ignoring corrupt pickle file - {file_name}")   
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(str(e))
            print(f"ignoring corrupt pickle file - {file_name}")  
    
    def dataset_length(self, type="identity"):
        if "all" == type:
            return len(self.all_y_cartesian_grasp)
        return len(self.y_cartesian_grasp) 

    def get_grasp_datapoint(self, i, type="identity"):
        if "all" == type:
            return self.all_img_grasp_embeddings[i], \
                    self.all_y_cartesian_grasp_normalized[i], \
                    self.all_y_cartesian_grasp[i]
        else:
            return self.img_grasp_embeddings[type][i], \
                    self.y_cartesian_grasp_normalized[i], \
                    self.y_cartesian_grasp[i]

    def get_release_datapoint(self, i, type="identity"):
        if "all" == type:
            return self.all_img_release_embeddings[i], \
                    self.all_y_cartesian_release_normalized[i], \
                    self.all_y_cartesian_release[i]
        else:
            return self.img_release_embeddings[type][i], \
                    self.y_cartesian_release_normalized[i], \
                    self.y_cartesian_release[i]

def parse_image_data(dir_path, filter_tag, grasp_release_dataset):
    for root, dirs, files in os.walk(dir_path):
        for d in dirs:
            parse_image_data(d, filter_tag, grasp_release_dataset)
        for f in files:
            abspath = os.path.join(root, f) 
            if not f.startswith("compressed") or not \
                f.endswith(".pickle") or not filter_tag in f:
                continue
            grasp_release_dataset.add_image_data(abspath)

def process_image_data(dir_paths, filter_tag, save_file_name=None):
    grasp_release_dataset = GraspReleaseDataset()
    for dir_path in dir_paths:
        parse_image_data(dir_path, filter_tag, grasp_release_dataset)
    if not grasp_release_dataset.validate_dataset():
        raise Exception("corrupt data set, cannot procede")
    print(f"grasp_release_dataset {filter_tag} - len {grasp_release_dataset.dataset_length()}")
    if save_file_name:
        grasp_release_dataset.save_dataset(save_file_name)
    return grasp_release_dataset

def parse_data(dir_path, filter_tag, grasp_release_dataset):
    for root, dirs, files in os.walk(dir_path):
        for d in dirs:
            parse_data(d, filter_tag, grasp_release_dataset)
        for f in files:
            abspath = os.path.join(root, f) 
            if not f.startswith("embed") or \
                not f.endswith(".pickle") or not filter_tag in f:
                continue
            grasp_release_dataset.add_data(abspath)

def process_data(dir_paths, filter_tag, save_file_name=None):
    grasp_release_dataset = GraspReleaseDataset()
    for dir_path in dir_paths:
        parse_data(dir_path, filter_tag, grasp_release_dataset)
    if not grasp_release_dataset.validate_dataset():
        raise Exception("corrupt data set, cannot procede")
    print(f"grasp_release_dataset {filter_tag} - len {grasp_release_dataset.dataset_length()}")
    if save_file_name:
        grasp_release_dataset.save_dataset(save_file_name)
    return grasp_release_dataset

def gen_datasets():
    dir_paths = [] 
    dir_paths.append("/mnt/nfs2/giriman/data/random_grasp")
    dir_paths.append("/mnt/nfs2/giriman/data/img_pri_sample")
    tags = ["robopen02", "robopen03", "robopen04", "robopen05"]
    for tag in tags:
        dataset_file_name = f"{tag}_clp_dataset.pickle"
        os.makedirs(TRAINING_DATASET_DIR, exist_ok=True)        
        full_dataset_file_name = os.path.join(TRAINING_DATASET_DIR, dataset_file_name)
        process_image_data(dir_paths, tag, full_dataset_file_name)

if __name__ == '__main__':
    gen_datasets()
