from pathlib import Path
from typing import Union, Literal

from tclogger import logger, shell_cmd
from tqdm import tqdm

from configs.envs import VIDEOS_ROOT


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


if __name__ == "__main__":
    mid = 946974
    videos_dir = Path(VIDEOS_ROOT) / str(mid)
    videos_paths = sorted(list(videos_dir.glob("*.mp4")), key=lambda x: x.name)
    converter = VideoToAudioConverter()
    for video_path in tqdm(videos_paths):
        converter.convert(video_path)

    # python -m converters.video_to_audio
