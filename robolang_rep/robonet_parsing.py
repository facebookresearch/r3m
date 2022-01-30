import math
import shutil
import tempfile
from decimal import Decimal
from fractions import Fraction
from typing import Iterable, List, Union, Tuple, Optional

import av
import numpy as np


##############################################################################################
## The code below parses Ego4D into frames for faster loading
##############################################################################################
import json
import os, random
# Load Data
print("RoboNet")
from robonet.datasets import load_metadata, RoboNetDataset
from torch.utils.data import DataLoader
metadata = load_metadata(args.path)
hparams = {'T': 30, 'action_mismatch': 3, 'state_mismatch': 3, 'splits':[0.8, 0.1, 0.1], 'normalize_images':False}
dataset = RoboNetDataset(metadata, "train", hparams)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=16)


images, states, actions = next(iter(loader))
images = np.transpose(images.numpy(), (0, 1, 3, 4, 2))
images *= 255
images = images.astype(np.uint8)

print(images.shape)
import imageio
writer = imageio.get_writer('test_frames.gif')
for t in range(images.shape[1]):
    writer.append_data(np.concatenate([b for b in images[:, t]], axis=-2))
writer.close()           

assert(False)
import pandas as pd
import time
paths = []
txts = []
lens = []
clips = 0
totalvids = 0

from multiprocessing import Pool
for vid in list(data['video_data'].keys()):
    for interval in data['video_data'][vid]["annotated_intervals"]:
        for action in interval["narrated_actions"]:
            try:
                t0 = time.time()
                pre = int(action['critical_frames']['pre_frame'])
                post = int(action['critical_frames']['post_frame'])
                txt = action['narration_text']
                txtls = txt.split(" ")
                txtlen = len(txtls)
                assert(txtlen < 50)
                assert((post-pre) > 5)
                txtls = [t for t in txtls if "#" not in t]
                txt = " ".join(txtls)
                os.makedirs(f"/private/home/surajn/data/ego4d/{vid}/{pre}_{post}/", exist_ok=True)
                with av.open(f"/datasets01/ego4d_track2/full_scale/v1/071621/{vid}", mode="r") as in_container:
                    in_video_stream = in_container.streams.video[0]
                    original_fps = in_video_stream.average_rate
                    frames_iterator = _get_frames(
                        range(pre,post), in_container, False, audio_buffer_frames=original_fps // 30
                    )
                    for i, frame in enumerate(frames_iterator):
                        img = frame.to_ndarray(format="rgb24")
                        im = preprocess(torch.Tensor(img).permute(2, 0, 1)).to(torch.uint8)
                        torchvision.io.write_jpeg(im, f"/private/home/surajn/data/ego4d/{vid}/{pre}_{post}/{i:06}.jpg")
                vidlen = i
                paths.append(f"/private/home/surajn/data/ego4d/{vid}/{pre}_{post}/")
                txts.append(txt)
                lens.append(vidlen)
                clips += 1
                print(totalvids, clips, txt, time.time() - t0)
            except:
                pass
    totalvids += 1
dc = {"path": paths, "len": lens, "txt": txts}
df = pd.DataFrame(dc)
df.to_csv("/private/home/surajn/data/ego4d/manifest.csv")
##############################################################################################


