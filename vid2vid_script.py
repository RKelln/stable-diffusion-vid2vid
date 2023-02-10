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

from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import modules.images as images
import modules.scripts as scripts
import numpy as np
from modules import processing
from modules.images import sanitize_filename_part
from modules.processing import Processed, get_fixed_seed, process_images
from modules.shared import state
from PIL import Image

# handle local imports
import sys, os
scripts_path = str(Path(os.getcwd(), 'scripts'))
print(os.getcwd(), sys.path)
if not scripts_path in sys.path:
    sys.path.append(scripts_path)
from vid2vid_helpers.vid2vid_schedules import *
from vid2vid_helpers.vid2vid_video import Video

DEFAULT_RESULT_PATH = "output"
DEFAULT_FRAMES_PATH = "input"


def fix_seed(s : Any) -> int:
    if s is None:
        return get_fixed_seed(s)
    
    return get_fixed_seed(int(s))


class Script(scripts.Script):
    def title(self):
        return "Vid2vid with schedules"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        input_path = gr.Video(label="Input video")

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
            keep = gr.Checkbox(label='Keep generated pngs?', value=False)

        with gr.Row():
            seed_schedule = gr.Textbox(label="Seed schedule", lines=1, 
                description="Relative to start. Format: <frame or time>: <seed>, ... where frame is an integer or a time in mm:ss and seed is an integer or -1 for random")
            denoise_schedule = gr.Textbox(label="Denoise schedule", lines=1, 
                description="Relative to start. Format: <frame or time>: <value>, ... where frame is an integer or a time in mm:ss and value from 0 to 1")
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
            denoise_schedule,
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
        denoise_schedule,
        save_video,
        output_crf,
        output_fps
    ):
        processing.fix_seed(p)
        initial_seed = p.seed
        initial_denoise = p.denoising_strength
        initial_info = None

        # set up paths
        input_path = Path(input_path.strip())
        if not Path.exists(input_path):
            raise RuntimeError(f"Input video does not exist: {input_path}")
        save_dir = Path(save_dir.strip())
        run_name = sanitize_filename_part(input_path.stem[:15] + "+" + p.prompt[:30])
        output_dir = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{run_name}"
        output_path = Path(save_dir, output_dir)
        frames_path = output_path / DEFAULT_FRAMES_PATH
        result_path = output_path / DEFAULT_RESULT_PATH
        Path.mkdir(frames_path, parents = True, exist_ok = True)
        Path.mkdir(result_path, parents = True, exist_ok = True)

        # video statistics
        video_fps = Video.fps(input_path)
        video_duration = Video.duration(input_path)

        start_time = parse_to_seconds(start, video_fps, video_duration)
        end_time = parse_to_seconds(end, video_fps, video_duration) # will be 0 if unspecified
        output_crf = int(output_crf)
        output_fps = float(output_fps)

        # save settings
        settings = f"""
{input_path}
{output_path}
Video:
Length: {video_duration}
FPS: {video_fps}
Start: {start_time} sec
End: {end_time} sec
Settings:
Seed schedule: {seed_schedule}
Denoise schedule: {denoise_schedule}
Output CRF: {output_crf}
Output FPS: {output_fps}
Diffusion values:
Sampler: {p.sampler_name}
Steps: {p.steps}
Width: {p.width}
Height: {p.height}
Prompt:
{p.prompt}
Neg:
{p.negative_prompt}
"""
        with open(output_path / "settings.txt", "w") as text_file:
                text_file.write(settings)

        # extract frames from input video
        Video.to_frames(input_path.stem, input_path, frames_path, extract_fps, p.width, p.height, start_time, end_time)

        # count extracted images
        frames = sorted(frames_path.glob(f"{input_path.stem}*.png"))

        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.subseed_strength = 0
        # Force Batch Count and Batch Size to 1
        p.batch_count = 1
        p.n_iter = 1

        # seeds: a list of sorted time and seed pairs
        seed_schedule = [(t, fix_seed(s)) for t, s  in parse_schedule(seed_schedule, video_fps, video_duration, initial_seed)]
        #print("seed_schedule: ", seed_schedule)
        seeds = seed_travel_planning(seed_schedule, len(frames), extract_fps, initial_seed)
        #print("seeds: ", seeds)

        # denoise
        denoise_schedule = [(t, float(s)) for t, s  in parse_schedule(denoise_schedule, video_fps, video_duration, initial_denoise)]
        #print(denoise_schedule)
        denoise = denoise_travel_planning(denoise_schedule, len(frames), extract_fps, initial_denoise)
        #print(denoise)

        if len(frames) != len(seeds) or len(seeds) != len(denoise):
            raise RuntimeError(f"Frame count ({len(frames)}) doesn't match seed ({len(seeds)}) or denoise ({len(denoise)}) count")

        # TODO: handle batch size > 1
        p.batch_size = 1
        # p.seed = [seed for _ in batch]
        # p.init_images = batch
        # batch = []

        state.job_count = len(frames)
        for i in range(len(frames)):
            if state.interrupted:
                break
            frame = i + 1
            state.job = f"Frame {frame}/{len(frames)}"
            p.seed, p.subseed, p.subseed_strength = seeds[i]
            p.denoising_strength = denoise[i]
            p.init_images = [Image.open(frames[i])]
            #print(f"{frames[i]}: Seed: {p.seed} subseed: {p.subseed} str: {p.subseed_strength:0.2f}, denoise: {p.denoising_strength:0.2f}")
            proc = process_images(p)
            if initial_info is None:
                initial_info = proc.info

            for output in proc.images:
                # save_image(image, path, basename, seed=None, prompt=None, extension='png', info=None, short_filename=False, no_prompt=False, grid=False, pnginfo_section_name='parameters', p=None, existing_info=None, forced_filename=None, suffix="", save_to_dirs=None):
                filename = f"{run_name}_{i:05}"
                images.save_image(output, result_path, "", info=proc.info, forced_filename=filename)
                #output.save(result_path / f"{filename}.png")
                #print("Saved: ", str(result_path / f"{filename}.png"))

        if save_video:
            Video.from_frames(run_name, output_path, result_path, output_fps, p.width, p.height, output_crf)

        if not keep:
            # delete images
            for f in result_path.glob(f"{run_name}*.png"):
                f.unlink()

        return Processed(p, [], p.seed, initial_info)

