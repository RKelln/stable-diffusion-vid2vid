# Vid2Vid with schedules
## for Automatic1111 webui

Video-2-video plugin that includes support for seed and denoise value travel.

Intended for use with https://github.com/AUTOMATIC1111/stable-diffusion-webui.
Save all files into `scripts` folder then restart the UI. (I.E. copy the `vid2vid_helpers` folder into `scripts` as well.)

Requires `ffmpeg` to be installed.

_**Please note**: this is a early version, with minimal testing, only tested with Linux. Expect bugs!_

## Credits
Filarius, orcist1, Leonm99, yownas

- https://github.com/Filarius
- https://github.com/Leonm99/Stable-Diffusion-Video2Video
- https://github.com/yownas/seed_travel


# Install

Install [ffmpeg](https://ffmpeg.org/) if it is not already installed.

Copy `vid2vid_script.py` and the `vid2vid_helpers` folder into the `scripts` folder.

    scripts/
        vid2vid_script.py
        vid2vid_helpers/
            vid2vid_schedules.py
            vid2vid_video.py


# Usage

On the `img2img` tab, set the regular diffusion parameters and select `Vid2vid with schedules` in the Script dropdown.

## Schedule format

Schedules are commas separated lists of time and value pairs separated by colons.

Time can be indicated by `hh:mm:ss.millisecond` format (e.g. `00:01:23.456`). Hours and minutes are optional.
Time can also be specified in seconds (e.g. `3.45`).
Time can also be specified by percentage (e.g. `25%`).
Otherwise the frame number can be used.
Times and frame numbers are relative to the start time specified.

### Seeds

Seed are specified by integers or by `-1` for a random seed. They may be surrounded by brackets but that not necessary.

There is a linear interpolation between all different seed values (using subseeds), so if you want to remain on the same seed for a period of time, set it to the same seed value at the start and end of that period.

Example:
`0:23456, 10%:99, 1:00:99, 90.3:-1, 100%:-1`


## Denoise values

Denoise values are float values from `0.0` to `1.0`, generally useful between `0.4` and `0.8`.

There is a linear interpolation between all different denoise values, so if you want to remain on the same value for a period of time, set it to the same value at the start and end of that period.

Example:
`0:0.2, 10%:0.6, 1:00:0.6, 90.3:0.7, 100%:0.8`