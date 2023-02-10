import subprocess
from pathlib import Path


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
