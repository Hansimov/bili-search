import json

from pathlib import Path
from tclogger import logger
from tqdm import tqdm
from typing import Union

from configs.envs import BILI_DATA_ROOT, LLMS_ENVS
from networks.llm_client import LLMClient


class SentencePuncter:
    def __init__(self):
        self.load_llm_client()

    def load_llm_client(self):
        llm_env = None
        for env in LLMS_ENVS:
            if "punct" in env["tasks"]:
                llm_env = env
                break
        if not llm_env:
            raise ValueError("× No valid llm client found for punct task!")
        self.llm_client = LLMClient(
            endpoint=llm_env["endpoint"],
            api_key=llm_env["api_key"],
            response_format=llm_env["format"],
        )
        self.llm_env = llm_env

    def convert(self, sentence: str):
        messages = [
            {
                "role": "user",
                "content": f'请给这句中文加上逗号和句号："{sentence}"',
            }
        ]
        self.llm_client.chat(
            messages=messages,
            model=self.llm_env["model"],
            temperature=0,
            seed=42,
            stream=False,
        )


class SubtitlePuncter:
    def generate_punct_path(self, punct_path: Union[str, Path] = None):
        if punct_path is None:
            self.punct_path = Path(self.subtitle_path).parent / (
                Path(self.subtitle_path).stem + "_puncted.json"
            )
        else:
            self.punct_path = Path(punct_path)
        return self.punct_path

    def convert(
        self,
        subtitle_path: Union[str, Path],
        punct_path: Union[str, Path] = None,
        overwrite: bool = False,
        verbose: bool = False,
    ):
        logger.enter_quiet(not verbose)

        self.subtitle_path = Path(subtitle_path)
        self.generate_punct_path(punct_path)

        logger.note(f"> Puncting subtitle:")
        logger.file(f"  - subtitle: [{self.subtitle_path}]")
        logger.file(f"  - puncted : [{self.punct_path}]")

        if not overwrite and self.punct_path.exists():
            logger.success(f"  + puncted subtitle existed: [{self.punct_path}]")
        else:
            with open(self.subtitle_path, "r", encoding="utf-8") as rf:
                subtitles = json.load(rf)

        logger.exit_quiet(not verbose)


if __name__ == "__main__":
    # mid = 946974
    # video_subtitles_dir = Path(BILI_DATA_ROOT) / str(mid) / "video_subtitles"
    # video_subtitle_paths = [
    #     x for x in video_subtitles_dir.glob("*.json") if not x.name.endswith("_puncted")
    # ]
    # video_subtitles_paths = sorted(video_subtitle_paths, key=lambda x: x.name)
    # puncter = SubtitlePuncter()
    # for video_subtitle_path in tqdm(video_subtitles_paths):
    #     puncter.convert(video_subtitle_path, verbose=True)

    sentence_punter = SentencePuncter()
    sentence_punter.convert(
        # "比如说平台可能希望的是一个收益然后上架的话可能是一个质量或者一些成本的保证对用户来说好像是满足他的诉求以满足他的体验"
        "质量或者一些成本的保证对用户来说好像是满足他的诉求以满足他的体验这是何为最佳匹配第二的话就是定义了匹配之后那么背后如何去做到这种精心化的匹配"
    )

    # python -m converters.audio.subtitle_puncter
