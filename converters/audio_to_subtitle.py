import json
import whisperx
import zhconv

from pathlib import Path
from tclogger import logger
from termcolor import colored
from tqdm import tqdm
from typing import Union, Literal, List

from configs.envs import BILI_DATA_ROOT
from converters.times import decimal_seconds_to_srt_timestamp


class WhisperX:
    def __init__(
        self,
        model_name: Literal[
            "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
        ] = "large-v3",
        language: Literal["en", "zh"] = "zh",
        device: Literal["cpu", "cuda"] = "cuda",
        device_index: int = 1,
        compute_type: Literal["float16", "int8"] = "float16",
        initial_prompt: str = None,
        chunk_size: float = 10,
    ):
        self.transcribe_model = None
        self.model_name = model_name
        self.language = language
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type
        self.initial_prompt = initial_prompt
        self.chunk_size = chunk_size

    def load_model(self):
        # if self.language in ["zh"]:
        #     self.initial_prompt = "以下是中文普通话句子。"

        self.asr_options = {
            "initial_prompt": self.initial_prompt,
        }

        logger.note(f"> Load model:", end=" ")
        logger.mesg(
            f"model=[{colored(self.model_name,'light_green')}], "
            f"device=[{colored(self.device,'light_green')}], "
            f"lang=[{colored(self.language,'light_green')}], "
            f"chunk_size=[{colored(self.chunk_size,'light_green')}s]"
        )
        self.transcribe_model = whisperx.load_model(
            self.model_name,
            device=self.device,
            device_index=self.device_index,
            compute_type=self.compute_type,
            language=self.language,
            asr_options=self.asr_options,
        )
        self.align_model, self.align_model_metadata = whisperx.load_align_model(
            language_code=self.language, device=self.device
        )

    def load_audio(self, audio_path: Union[str, Path]):
        self.audio = whisperx.load_audio(str(audio_path))

    def transcribe(self):
        logger.note("> Transcribing ...")
        self.transcibe_result = self.transcribe_model.transcribe(
            audio=self.audio,
            chunk_size=self.chunk_size,
            language=self.language,
            print_progress=False,
        )

    def align(self):
        logger.note("> Aligning ...")
        self.align_result = whisperx.align(
            self.transcibe_result["segments"],
            model=self.align_model,
            align_model_metadata=self.align_model_metadata,
            audio=self.audio,
            device=self.device,
            print_progress=False,
        )

    def save_to_file(self, subtitle_path: Union[str, Path]):
        logger.note(f"> Subtitle saved to:")
        logger.file(f"  - [{subtitle_path}]")

        output_suffix = subtitle_path.suffix

        subtitle_lines = []
        subtitle_json_path = subtitle_path.with_suffix(".json")

        res_str = ""
        for idx, seg in enumerate(self.align_result["segments"]):
            start_ts = decimal_seconds_to_srt_timestamp(seg["start"])
            end_ts = decimal_seconds_to_srt_timestamp(seg["end"])
            text = seg["text"]

            if self.language in ["zh"]:
                text = zhconv.convert(text, "zh-cn")

            subtitle_line = {
                "start_seconds": seg["start"],
                "end_seconds": seg["end"],
                "start_ts": start_ts,
                "end_ts": end_ts,
                "text": text,
            }
            subtitle_lines.append(subtitle_line)

            if output_suffix == ".srt":
                segment_str = f"{idx+1}\n" f"{start_ts} --> {end_ts}\n" f"{text}\n\n"
            else:
                segment_str = f"{seg['text']}\n"

            res_str += segment_str

        with open(subtitle_json_path, "w") as wf:
            json.dump(subtitle_lines, wf, ensure_ascii=False, indent=False)

        with open(subtitle_path, "w") as wf:
            wf.write(res_str)

    def convert(self, audio_path: Union[str, Path], subtitle_path: Union[str, Path]):
        if self.transcribe_model is None:
            self.load_model()
        self.load_audio(audio_path)
        self.transcribe()
        self.align()
        self.save_to_file(subtitle_path)

