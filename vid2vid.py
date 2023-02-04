# Vid2vid with schedules (seed travel, etc)
#
# Intended for use with https://github.com/AUTOMATIC1111/stable-diffusion-webui
# Save this in script folder then restart the UI.
#
# Authors: 
#  RKelln, Filarius, orcist1, Leonm99, yownas
# Inspired from:
# - https://github.com/Filarius
# - https://github.com/Leonm99/Stable-Diffusion-Video2Video
# - https://github.com/yownas/seed_travel


import os
import random
import sys
from datetime import datetime
from pathlib import Path
import subprocess

import gradio as gr
import modules.scripts as scripts
import numpy as np
from modules import processing
from modules.images import sanitize_filename_part
from modules.processing import Processed, process_images
from modules.shared import state
from PIL import Image

MAX_SEED = 2147483647

# parse a string time in the form of 00:00:00.000 to seconds
def time_string_to_seconds(time_str : str) -> float:
    time_parts = time_str.strip().split(':')
    if len(time_parts) == 0:
        return float(time_str)
    
    h = m = s = s_ms = 0.0
    if len(time_parts) == 1:
        s_ms = float(time_parts[0])
    elif len(time_parts) == 2:
        m, s_ms = map(float, time_parts)
    else:
        h, m, s_ms = map(float, time_parts)
    s, ms = map(float, str(s_ms).split('.'))
    return 3600*h + 60*m + s + ms/1000.0

# given a string of time or frame, return time in seconds
def parse_to_seconds(s : str, fps : float, duration : float) -> float:
    if fps <= 0:
        raise RuntimeError(f"FPS must be > 0: {fps}")
    if duration <= 0:
        raise RuntimeError(f"Duration must be > 0: {duration}")

    s = s.strip()
    if s == "":
        return 0.
    elif s.endswith('%'):
        percentage = float(s.strip('%'))
        return (percentage / 100.) * duration
    elif ':' in s:
        return min(duration, time_string_to_seconds(s))
    elif '.' in s:
        return min(duration, float(s))
    else:
        return min(duration, float(s) / fps)


def parse_schedule(s : str, fps : float, duration : float) -> list:
    seeds = []
    if s != "":
        seed_strs = [x.strip() for x in s.split(",")]
        for seed_str in seed_strs:
            time_seed_split = seed_str.rsplit(":", maxsplit=1)
            t = parse_to_seconds(time_seed_split[0], fps, duration)
            seed = int(time_seed_split[1].strip(" ()"))
            if seed < 0:
                seed = random_seed()
            seeds.append((t, seed))
    if len(seeds) == 0:
        seeds.append((0,random_seed()))
    return sorted(seeds)


def time_to_frame(time : float, fps : float) -> int:
    # compensate for floats being weird
    # at 30 fps, first frame from 0 to 32ms, second frame starts at 33ms
    # e.g. 333ms = 0.333 * 30 fps = 9.99, we want to return frame 10
    return int( (time * fps) + 0.01 )


def random_seed() -> int:
    return random.randint(0, MAX_SEED)


def seed_travel_planning(seeds : list, frame_count : int, fps : float, starting_seed : int) -> list:
    seeds_per_frame = []
    current_frame = frame = 0
    seed = next_seed = starting_seed
    next_frame = frame_count

    # NOTE: frames count from 1, but list is 0 indexed
    while len(seeds_per_frame) < frame_count:
        current_frame = len(seeds_per_frame) + 1
        next_frame = frame_count # the end
        if len(seeds) > 0:
            # set next_frame and next_seed here to transition from starting seed
            time, next_seed = seeds[0] 
            next_frame = frame = time_to_frame(time, fps)
            if current_frame >= frame:
                _, seed = seeds.pop(0)
                if len(seeds) > 0:
                    next_time, next_seed = seeds[0]
                    next_frame = time_to_frame(next_time, fps)
                else:
                    next_seed = -1
                    next_frame = frame_count + 1
        steps = max(1, next_frame - current_frame)
        for i in range(current_frame, next_frame):
            if next_seed == -1 or next_seed == seed:
                seeds_per_frame.append((seed, 0, 0))
            else:
                strength = float(i-current_frame)/float(steps)
                if strength == 0.0:
                    seeds_per_frame.append((seed, 0, 0))
                elif strength == 1.0: # shouldn't happen
                    seeds_per_frame.append((next_seed, 0, 0))
                else:
                    seeds_per_frame.append((seed, next_seed, strength))
    #print(seeds_per_frame)
    return seeds_per_frame


class Script(scripts.Script):
    def title(self):
        return "Vid2vid with schedules"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        input_path = gr.Video(label="Input video")
        # output_path = gr.Textbox(label="Output file path", lines=1)

        with gr.Row():
            extract_fps = gr.Slider(
                label="Extracted frames per second",
                minimum=1,
                maximum=60,
                step=1,
                value=15,
            )
            start = gr.Textbox(label="Start time", value="00:00:00", lines=1, description="Time (hh:mm:ss.ms) or seconds (float) or frame (integer), defaults to start")
            end = gr.Textbox(label="End time", value="00:00:00", lines=1, description="Time (hh:mm:ss.ms) or seconds (float) or frame (integer), defaults to end")

        with gr.Row():
            save_dir = gr.Textbox(label="Output file path", lines=1, value="outputs/img2img-video/vid2vid/")
            #video_frames_dir = gr.Textbox(label="Output file path", lines=1, value="outputs/img2img-video/vid2vid/temp_frames")
            keep = gr.Checkbox(label='Keep generated pngs?', value=False)

        with gr.Row():
            seed_schedule = gr.Textbox(label="Seed schedule", lines=1, description="Relative to start. Format: <frame or time>: <seed>, ... where frame is an integer or a time in mm:ss and seed in an integer or -1 for random")

        with gr.Row():
            save_video = gr.Checkbox(label='Save results as video', value=True)
            output_crf = gr.Slider(
                label="Video CRF (quality, less is better, x264 param)",
                minimum=16,
                maximum=32,
                step=1,
                value=24,
            )
            output_fps = gr.Slider(
                label="Video FPS",
                minimum=1,
                maximum=60,
                step=1,
                value=30,
            )

        return [
            input_path,
            extract_fps,
            save_dir,
            keep,
            start,
            end,
            seed_schedule,
            save_video,
            output_crf,
            output_fps,
        ]

    def run(
        self,
        p,
        input_path,
        extract_fps,
        save_dir,
        keep,
        start,
        end,
        seed_schedule,
        save_video,
        output_crf,
        output_fps
    ):
        print(f"input_path: {input_path}, extract_fps: {extract_fps}, save_dir: {save_dir}")
       
        processing.fix_seed(p)

        initial_seed = p.seed
        initial_info = None

        # set up paths
        input_path = Path(input_path.strip())
        if not Path.exists(input_path):
            raise RuntimeError(f"Input video does not exist: {input_path}")
        save_dir = Path(save_dir.strip())
        run_name = sanitize_filename_part(p.prompt[:50])
        output_dir = f"{datetime.now():%Y-%m-%d-%H-%M-%S}_{run_name}"
        output_path = Path(save_dir, output_dir)
        Path.mkdir(output_path, parents = True, exist_ok = True)
        print("Output: ", output_path)

        # video statistics
        video_fps = Video.fps(input_path)
        video_duration = Video.duration(input_path)

        start_time = parse_to_seconds(start, video_fps, video_duration)
        end_time = parse_to_seconds(end, video_fps, video_duration) # will be 0 if unspecified
        output_crf = int(output_crf)
        output_fps = float(output_fps)

        # extract frames from input video
        Video.to_frames(input_path.stem, input_path, output_path, extract_fps, p.width, p.height, start_time, end_time)

        # count extracted images
        frames = sorted(output_path.glob(f"{input_path.stem}*.png"))
        
        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.subseed_strength = 0
        # Force Batch Count and Batch Size to 1
        p.batch_count = 1
        p.n_iter = 1

        # seeds: a list of sorted time and seed pairs
        seeds = []
        if seed_schedule != "":
            seed_schedule = parse_schedule(seed_schedule, video_fps, video_duration)
            #print(seed_schedule)
            seeds = seed_travel_planning(seed_schedule,len(frames), video_fps, initial_seed)

        state.job_count = len(frames)

        # TODO: handle batch size > 1
        p.batch_size = 1
        # p.seed = [seed for _ in batch]
        # p.init_images = batch
        # batch = []

        for i in range(len(frames)):
            if state.interrupted:
                break
            frame = i + 1
            state.job = f"Frame {frame}/{len(frames)}"
            p.seed, p.subseed, p.subseed_strength = seeds[i]
            p.init_images = [Image.open(frames[i])]
            proc = process_images(p)
            if initial_info is None:
                initial_info = proc.info

            for output in proc.images:
                output.save(output_path / f"{run_name}_{i:05}.png")
                print("Saved: ", str(output_path / f"{run_name}_{i:05}.png"))

        if save_video:
            Video.from_frames(run_name, output_path, output_path, output_fps, p.width, p.height, output_crf)

        if not keep:
            # delete images
            for f in output_path.glob(f"{run_name}*.png"):
                f.unlink()

        return Processed(p, [], p.seed, initial_info)


class Video:

    def duration(video_path : str) -> float:
        result = subprocess.run(
            [
                "ffprobe",
                "-v","error",
                "-select_streams","v:0",
                "-of","default=noprint_wrappers=1:nokey=1",
                "-show_entries", "stream=duration",
                str(video_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            print(result.stderr.decode('utf-8'))
            raise RuntimeError(result.stderr.decode('utf-8'))
        duration = float(result.stdout.decode('utf-8'))
        return duration

    def fps(video_path : str) -> float:
        result = subprocess.run(
            [
                "ffprobe",
                "-v","error",
                "-select_streams","v:0",
                "-of","default=noprint_wrappers=1:nokey=1",
                "-show_entries", "stream=r_frame_rate",
                str(video_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            print(result.stderr.decode('utf-8'))
            raise RuntimeError(result.stderr.decode('utf-8'))
        result_string = result.stdout.decode('utf-8').split()[0].split('/')
        fps = float(result_string[0])/float(result_string[1])
        return fps


    @staticmethod
    def to_frames(filename : str, video_path : str, output_path : str, fps : int, width : int, height : int, start_time : float = 0, end_time : float = 0):

        image_path = Path(output_path, f"{filename}_%05d.png")

        cmd = [
            'ffmpeg',
            '-y',
            '-loglevel','panic',
        ]
        if start_time > 0:
            cmd += ['-ss', str(start_time)]
        if end_time > 0:
            cmd += ['-to', str(end_time)]
        cmd += [
            '-i', str(video_path),
            '-s:v', f"{width}x{height}",
            '-r', str(fps),
            '-vsync', '1',
            str(image_path)
        ]
        #print(" ".join(cmd))

        result = subprocess.run(cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # if result.stdout is not None:
        #     print('stdout: ', result.stdout.decode('utf-8'))
        # if result.stderr is not None:
        #     print('stderr: ', result.stderr.decode('utf-8'))
        if result.returncode != 0:
            print(result.stderr.decode('utf-8'))
            raise RuntimeError(result.stderr.decode('utf-8'))

    @staticmethod
    def from_frames(filename : str, output_path : str, input_path : str, fps : float, width : int, height: int, crf : int = 24) -> str:
        
        image_path = Path(input_path, f"{str(filename)}_%05d.png")
        video_path = Path(output_path, f"{str(filename)}.mp4")

        cmd = [
            'ffmpeg',
            '-y',
            '-loglevel', 'panic',
            '-vcodec', 'png',
            '-i', str(image_path),
            '-r', str(fps),
            '-s:v', f"{width}x{height}",
            '-pix_fmt', 'rgb24',
            '-c:v','libx264',
            '-crf', str(crf),
            str(video_path),
        ]
        #print(" ".join(cmd))

        result = subprocess.run(cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # if result.stdout is not None:
        #     print('stdout: ', result.stdout.decode('utf-8'))
        # if result.stderr is not None:
        #     print('stderr: ', result.stderr.decode('utf-8'))
        if result.returncode != 0:
            print(result.stderr.decode('utf-8'))
            raise RuntimeError(result.stderr.decode('utf-8'))

        return video_path