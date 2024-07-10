import argparse
import concurrent.futures
import sys

from pathlib import Path
from tclogger import logger, shell_cmd
from tqdm import tqdm
from typing import Union, Literal, List

from configs.envs import BILI_DATA_ROOT


class VideoToAudioConverter:
    def __init__(self):
        self.ffmpeg = "ffmpeg"
        self.cmd_args = '-y -i "{}" -vn "{}"'

    def generate_audio_path(
        self,
        audio_path: Union[str, Path] = None,
        suffix: Literal[".mp3", ".wav"] = ".mp3",
    ):
        if audio_path is None:
            self.audio_path = Path(self.video_path).with_suffix(suffix)
        else:
            self.audio_path = Path(audio_path)
        return self.audio_path

    def is_audio_exists(self):
        if self.audio_path.exists():
            return True
        return False

    def convert(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path] = None,
        overwrite: bool = False,
        verbose: bool = False,
    ):
        logger.enter_quiet(not verbose)

        self.video_path = Path(video_path)
        self.generate_audio_path(audio_path)

        cmd_args = self.cmd_args.format(self.video_path, self.audio_path)
        if not verbose:
            cmd_args = f"-loglevel error {cmd_args}"
        cmd_str = f"{self.ffmpeg} {cmd_args}"

        logger.note(f"> Convert video to audio:")
        logger.file(f"  - video: [{self.video_path}]")
        logger.file(f"  - audio: [{self.audio_path}]")

        if not overwrite and self.is_audio_exists():
            logger.success(f"  + audio existed: [{self.audio_path}]")
        else:
            shell_cmd(cmd_str)

        logger.exit_quiet(not verbose)

    def bactch_convert(
        self,
        video_paths: List[Union[str, Path]],
        audio_paths: List[Union[str, Path]] = None,
        overwrite: bool = False,
        verbose: bool = False,
    ):
        if audio_paths is None:
            audio_paths = [None] * len(video_paths)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.convert, video_path, audio_path, overwrite, verbose
                )
                for video_path, audio_path in zip(video_paths, audio_paths)
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                future.result()


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ArgParser, self).__init__(*args, **kwargs)
        self.add_argument(
            "-m",
            "--mid",
            type=int,
            help="User mid",
        )
        self.add_argument(
            "-o",
            "--overwrite",
            action="store_true",
            help="Overwrite existed audios",
        )
        self.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Verbose",
        )

        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])


if __name__ == "__main__":
    args = ArgParser().args
    mid = args.mid or 946974
    videos_dir = Path(BILI_DATA_ROOT) / str(mid) / "videos"
    videos_paths = sorted(list(videos_dir.glob("*.mp4")), key=lambda x: x.name)
    converter = VideoToAudioConverter()
    converter.bactch_convert(
        videos_paths, overwrite=args.overwrite, verbose=args.verbose
    )

    # python -m converters.video_to_audio -m 14871346
    # python -m converters.video_to_audio -m 14871346 -o
    # python -m converters.video_to_audio -m 14871346 -o -v
