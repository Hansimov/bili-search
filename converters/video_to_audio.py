from pathlib import Path
from typing import Union, Literal
from tclogger import logger, shell_cmd
from configs.envs import VIDEOS_ROOT


class VideoToAudioConverter:
    def __init__(self):
        self.cmd = 'ffmpeg -y -i "{}" -vn "{}"'

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
    ):
        self.video_path = Path(video_path)
        self.generate_audio_path(audio_path)
        cmd_ffmpeg = self.cmd.format(self.video_path, self.audio_path)

        logger.note(f"> Convert video to audio:")
        logger.file(f"  - video: [{self.video_path}]")
        logger.file(f"  - audio: [{self.audio_path}]")

        if not overwrite and self.is_audio_exists():
            logger.success(f"  + audio existed: [{self.audio_path}]")
        else:
            shell_cmd(cmd_ffmpeg)


if __name__ == "__main__":
    mid = 946974
    videos_dir = Path(VIDEOS_ROOT) / str(mid)
    videos_paths = sorted(list(videos_dir.glob("*.mp4")), key=lambda x: x.name)
    converter = VideoToAudioConverter()
    for video_path in videos_paths[:1]:
        converter.convert(video_path)

    # python -m converters.video_to_audio
