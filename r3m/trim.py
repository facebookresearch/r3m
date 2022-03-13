import math
import shutil
import tempfile
from decimal import Decimal
from fractions import Fraction
from typing import Iterable, List, Union, Tuple, Optional

import av
import numpy as np
# from ego4d.datasetmanager.utils.ffmpeg_utils import offset_audio


def simple_frame_to_time(frame: int, fps: Fraction, start_pts: int) -> Fraction:
    """
    Assumption: start_pts == 0
    """
    if start_pts != 0:
        raise AssertionError("does not support start_pts != 0")
    return frame / fps


def simple_time_to_frame(
    time_sec: float,
    fps: Fraction,
    start_pts: int,
    require_exact_frame: bool = False,
) -> int:
    """
    Simpler method to time_to_quantize_pts_frame

    Can be used to convert a time difference to the number of frames

    Assumption: start_pts == 0

    Parameters
        time_sec
            Time in seconds
        fps
            fps of video
        start_pts
            starting presentation timestamp of video

            Must be 0

            Please use this parameter if you are deriving
        require_exact_frame
            Whether or not the given time must land on an exact frame
    """
    if start_pts != 0:
        raise AssertionError("does not current support start_pts != 0")

    if not isinstance(time_sec, float):
        raise AssertionError("time_sec should be float")

    frame = int(np.round(time_sec * fps))
    if require_exact_frame:
        frame_sec = frame / fps
        if abs(frame_sec - time_sec) >= 1e-6:
            raise AssertionError("given time does not land on an exact frame")
    return frame


def frames_to_select(
    start_frame: int,
    end_frame: int,
    original_fps: int,
    new_fps: int,
) -> Iterable[int]:
    # ensure the new fps is divisible by the old
    assert original_fps % new_fps == 0

    # check some obvious things
    assert end_frame >= start_frame

    num_frames = end_frame - start_frame + 1
    skip_number = original_fps // new_fps
    for i in range(0, num_frames, skip_number):
        yield i + start_frame


def time_to_quantize_pts_frame(
    time: Union[float, Decimal],
    fps: Fraction,
    base: Fraction,
    start_pt: int,
    inclusive: bool,
) -> Tuple[float, int]:
    """
    converts a timestamp to a video frame number and quantized time_pts of a cannonical video
    """
    assert type(start_pt) == int
    # pyre-fixme[58]: `//` is not supported for operand types `Union[Decimal,
    #  float]` and `Fraction`.
    pt = time // base
    diff = pts_difference_per_frame(fps, base)
    quantized_time = pt - start_pt

    # Here we do a ceiling on frame number to make sure we don't go before the 0th
    # frame with the 0.0 input time. If 'time' == 0.0 and 'start_pt' > 1, if we did not ceil
    # we could end up returning a -1 frame and a < 0.0 quantized time.
    frame = math.ceil(quantized_time / diff)

    timestamp_sec = pts_to_time_seconds(frame_index_to_pts(frame, start_pt, diff), base)
    return float(timestamp_sec), frame if inclusive else frame - 1


def pts_difference_per_frame(fps: Fraction, time_base: Fraction) -> int:
    r"""
    utility method to determine the difference between two consecutive video
    frames in pts, based off the fps and time base.
    """
    pt = (1 / fps) * (1 / time_base)
    assert pt.denominator == 1, "should be whole number"
    return int(pt)


def frame_index_to_pts(frame: int, start_pt: int, diff_per_frame: int) -> int:
    """
    given a frame number and a starting pt offset, compute the expected pt for the frame.

    Frame is assumed to be an index (0-based)
    """
    return start_pt + frame * diff_per_frame


def pts_to_time_seconds(pts: int, base: Fraction) -> Fraction:
    """
    converts a pt to the time (in seconds)

    returns:
        a Fraction (assuming the base is a Fraction)
    """
    return pts * base


def _get_frames_pts(
    video_pts_set: List[int],
    # pyre-fixme[11]: Annotation `Container` is not defined as a type.
    container: av.container.Container,
    include_audio: bool,
    include_additional_audio_pts: int,
    # pyre-fixme[11]: Annotation `Frame` is not defined as a type.
) -> Iterable[av.frame.Frame]:
    """
    Gets the video/audio frames from a container given:

    Inputs:
        video_pts_set
            the set of video pts to retrieve
        container
            the container to get the pts from
        include_audio
            Determines whether to ignore the audio stream in the first place
        include_additional_audio_pts
            Additional amount of time to include for audio frames.

            pts must be relative to video base
    """
    assert len(container.streams.video) == 1

    min_pts = min(video_pts_set)
    max_pts = max(video_pts_set)
    # pyre-fixme[9]: video_pts_set has type `List[int]`; used as `Set[int]`.
    video_pts_set = set(video_pts_set)  # for O(1) lookup

    video_stream = container.streams.video[0]
    fps: Fraction = video_stream.average_rate
    video_base: Fraction = video_stream.time_base
    video_pt_diff = pts_difference_per_frame(fps, video_base)

    # [start, end) time
    clip_start_sec = pts_to_time_seconds(min_pts, video_base)
    clip_end_sec = pts_to_time_seconds(max_pts, video_base)

    # add some additional time for audio packets
    clip_end_sec += max(
        pts_to_time_seconds(include_additional_audio_pts, video_base), 1 / fps
    )

    # --- setup
    streams_to_decode = {"video": 0}
    if (
        include_audio
        and container.streams.audio is not None
        and len(container.streams.audio) > 0
    ):
        assert len(container.streams.audio) == 1
        streams_to_decode["audio"] = 0
        audio_base: Fraction = container.streams.audio[0].time_base

    # seek to the point we need in the video
    # with some buffer room, just in-case the seek is not precise
    seek_pts = max(0, min_pts - 2 * video_pt_diff)
    video_stream.seek(seek_pts)
    if "audio" in streams_to_decode:
        assert len(container.streams.audio) == 1
        audio_stream = container.streams.audio[0]
        # pyre-fixme[61]: `audio_base` may not be initialized here.
        audio_seek_pts = int(seek_pts * video_base / audio_base)
        audio_stream.seek(audio_seek_pts)

    # --- iterate over video

    # used for validation
    previous_video_pts = None
    previous_audio_pts = None

    for frame in container.decode(**streams_to_decode):

        if isinstance(frame, av.AudioFrame):
            assert include_audio
            # ensure frames are in order
            assert previous_audio_pts is None or previous_audio_pts < frame.pts
            previous_audio_pts = frame.pts

            # pyre-fixme[61]: `audio_base` may not be initialized here.
            audio_time_sec = pts_to_time_seconds(frame.pts, audio_base)

            # we want all the audio frames in this region
            if audio_time_sec >= clip_start_sec and audio_time_sec < clip_end_sec:
                yield frame
            elif audio_time_sec >= clip_end_sec:
                break

        elif isinstance(frame, av.VideoFrame):
            video_time_sec = pts_to_time_seconds(frame.pts, video_base)
            if video_time_sec >= clip_end_sec:
                break

            # ensure frames are in order
            assert previous_video_pts is None or previous_video_pts < frame.pts

            if frame.pts in video_pts_set:
                # check that the frame is in range
                assert (
                    video_time_sec >= clip_start_sec and video_time_sec < clip_end_sec
                ), f"""
                video frame at time={video_time_sec} (pts={frame.pts})
                out of range for time [{clip_start_sec}, {clip_end_sec}]
                """

                yield frame


def _get_frames(
    video_frames: List[int],
    container: av.container.Container,
    include_audio: bool,
    audio_buffer_frames: int = 0,
) -> Iterable[av.frame.Frame]:
    assert len(container.streams.video) == 1

    video_stream = container.streams.video[0]
    video_start: int = video_stream.start_time
    video_base: Fraction = video_stream.time_base
    fps: Fraction = video_stream.average_rate
    video_pt_diff = pts_difference_per_frame(fps, video_base)

    audio_buffer_pts = (
        frame_index_to_pts(audio_buffer_frames, 0, video_pt_diff)
        if include_audio
        else 0
    )

    time_pts_set = [
        frame_index_to_pts(f, video_start, video_pt_diff) for f in video_frames
    ]
    return _get_frames_pts(time_pts_set, container, include_audio, audio_buffer_pts)


def _trim_video(
    new_fps: Union[int, Fraction],
    frames: List[int],
    in_container: av.container.Container,
    out_container: av.container.Container,
    codec: str,
    crf: int,
    include_audio: bool,
    is_reversed: bool,
) -> Tuple[bool, Optional[Fraction]]:
    """
    Method to trim video based off pyav containers

    returns:
        tuple of:
        - boolean to signify successfully encoded the specified number of frames
        - float for the amount the audio should be offset by

    NOTES:
    - PyAV cannot generate blank audio frames
    """
    video_stream = out_container.add_stream(codec, new_fps)
    in_video_stream = in_container.streams.video[0]
    video_stream.width = in_video_stream.width
    video_stream.height = in_video_stream.height
    video_stream.pix_fmt = in_video_stream.pix_fmt

    video_stream.options["crf"] = str(crf)

    # if the container has audio, add it to the output
    in_audio_stream = None
    audio_stream = None
    if (
        include_audio
        and in_container.streams.audio is not None
        and len(in_container.streams.audio) > 0
    ):
        assert len(in_container.streams.audio) == 1
        in_audio_stream = in_container.streams.audio[0]
        audio_stream = out_container.add_stream("aac", in_audio_stream.rate)

    out_container.start_encoding()

    original_fps = in_video_stream.average_rate

    v_frame = 0  # counter for validation
    first_video_pt_time = None
    first_audio_pt_time = None
    has_audio_frames = False

    frames_iterator = _get_frames(
        frames, in_container, include_audio, audio_buffer_frames=original_fps // new_fps
    )
    if is_reversed:
        assert (
            not include_audio
        ), "is_reversed = True - has not been tested with include_audio"
        assert len(frames) <= 100, (
            "is_reversed = True - too many frames "
            "remove me if you want to not limit this"
        )
        # NOTE: need to convert to list first
        frames_iterator = reversed(list(frames_iterator))

    for frame in frames_iterator:
        if isinstance(frame, av.AudioFrame):
            assert include_audio and audio_stream is not None

            if first_audio_pt_time is None:
                first_audio_pt_time = pts_to_time_seconds(
                    frame.pts, in_audio_stream.time_base
                )

            # reset the pt to remove errors
            frame.pts = None
            has_audio_frames = True

            for packet in audio_stream.encode(frame):
                out_container.mux(packet)
        elif isinstance(frame, av.VideoFrame):
            # copy the frame to not run into errors w.r.t pts
            if first_video_pt_time is None:
                first_video_pt_time = pts_to_time_seconds(
                    frame.pts, in_video_stream.time_base
                )

            img = frame.to_ndarray(format="rgb24")
            copy_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in video_stream.encode(copy_frame):
                out_container.mux(packet)
                v_frame += 1

    # flush the packets
    for packet in video_stream.encode():
        out_container.mux(packet)
        v_frame += 1

    # if doesn't include audio => exception will be thrown
    audio_offset_from_video_start_sec: Optional[Fraction] = None
    if audio_stream is not None and has_audio_frames:
        # and flush the audio frames
        for packet in audio_stream.encode():
            out_container.mux(packet)

        # pyre-fixme[58]: `-` is not supported for operand types
        #  `Optional[Fraction]` and `Optional[Fraction]`.
        audio_offset_from_video_start_sec = first_audio_pt_time - first_video_pt_time
        if audio_offset_from_video_start_sec == 0:
            audio_offset_from_video_start_sec = None
        else:
            assert audio_offset_from_video_start_sec > 0

    # verify we have the expected number of frames
    return v_frame == len(frames), audio_offset_from_video_start_sec


def trim_video(
    new_fps: Union[int, Fraction],
    frame_indices: List[int],
    in_path: str,
    out_path: str,
    codec: str = "libx264",
    crf: int = 28,
    include_audio: bool = True,
    is_reversed: bool = False,
) -> bool:
    print("Trimming clip")
    with av.open(in_path, mode="r") as in_container:
        with av.open(out_path, mode="w") as out_container:
            valid, offset_time = _trim_video(
                new_fps,
                frame_indices,
                in_container,
                out_container,
                codec,
                crf,
                include_audio,
                is_reversed,
            )

        # may need to offset the audio
        # if include_audio is False there won't be an offset time
        if offset_time is not None:
            assert include_audio
            with tempfile.TemporaryDirectory() as tempdir:
                temp_path = f"{tempdir}/offset.mp4"
                offset_audio(
                    in_path=out_path, out_path=temp_path, offset=float(offset_time)
                )
                shutil.copy(temp_path, out_path)

        return valid


##############################################################################################
## The code below parses Ego4D into frames for faster loading
##############################################################################################
import json
import os, random
f = open("/private/home/surajn/fho_release_v1_28-11-21.json",)
data = json.load(f)
from torchvision import transforms
import torchvision
import torch
preprocess = torch.nn.Sequential(
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                )

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