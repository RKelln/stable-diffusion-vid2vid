import subprocess
from pathlib import Path


class Video:

    @classmethod
    def duration(cls, video_path : str) -> float:
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


    @classmethod
    def fps(cls, video_path : str) -> float:
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


    @classmethod
    def handle_subprocess_result(cls, result):
        if result.returncode != 0:
            msgs = []
            if result.stdout is not None:
                msgs.append(result.stdout.decode('utf-8'))
            if result.stderr is not None:
                msgs.append(result.stderr.decode('utf-8'))
            msg = " ".join(msgs)
            print(msg)
            raise RuntimeError(msg)


    @classmethod
    def to_frames(cls, filename : str, video_path : str, output_path : str, fps : int, width : int, height : int, start_time : float = 0, end_time : float = 0, crop : bool = False):

        image_path = Path(output_path, f"{filename}_%05d.png")

        scale_params = ['-s:v', f"{width}x{height}"]
        if crop:
            scale_params = ['-vf', f"crop=in_h*{width}/{height}:in_h,scale=-2:{height}"]

        cmd = [
            'ffmpeg',
            '-y',
            '-loglevel','panic',
        ]
        if start_time > 0:
            cmd += ['-ss', str(start_time)]
        if end_time > 0:
            cmd += ['-to', str(end_time)]
        cmd += ['-i', str(video_path)]
        cmd += scale_params
        cmd += [
            '-r', str(fps),
            '-vsync', '1',
            str(image_path)
        ]
        print(" ".join(cmd))

        result = subprocess.run(cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        cls.handle_subprocess_result(result)


    @classmethod
    def from_frames(cls, filename : str, output_path : str, input_path : str, fps : float, width : int, height: int, crf : int = 24) -> str:
        
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
        cls.handle_subprocess_result(result)

        return video_path
