import ast
import json
import re
import requests

from tclogger import logger, Runtimer
from typing import Literal


class LLMClient:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_format: Literal["openai", "ollama"] = "openai",
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_format = api_format

    def create_response(
        self,
        messages: list,
        model: str,
        temperature: float = 0,
        seed: int = 42,
        stream: bool = True,
    ):
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        options = {
            "temperature": temperature,
            "seed": seed,
        }
        if self.api_format == "ollama":
            payload["options"] = options
        else:
            payload.update(options)

        response = requests.post(
            self.endpoint, headers=headers, json=payload, stream=stream
        )
        return response

    def parse_stream_response(self, response: requests.Response):
        response_content = ""
        for line in response.iter_lines():
            line = line.decode("utf-8")
            remove_patterns = [r"^\s*data:\s*", r"^\s*\[DONE\]\s*"]
            for pattern in remove_patterns:
                line = re.sub(pattern, "", line).strip()

            if line:
                try:
                    line_data = json.loads(line)
                except Exception as e:
                    try:
                        line_data = ast.literal_eval(line)
                    except:
                        logger.warn(f"× Error: {line}")
                        raise e

                if self.api_format == "ollama":
                    # https://github.com/ollama/ollama/blob/main/docs/api.md#response-9
                    delta_data = line_data["message"]
                    finish_reason = "stop" if line_data["done"] else None
                else:
                    # https://platform.openai.com/docs/api-reference/chat/streaming
                    delta_data = line_data["choices"][0]["delta"]
                    finish_reason = line_data["choices"][0]["finish_reason"]

                if "role" in delta_data:
                    role = delta_data["role"]
                if "content" in delta_data:
                    delta_content = delta_data["content"]
                    response_content += delta_content
                    logger.mesg(delta_content, end="")
                if finish_reason == "stop":
                    logger.success("\n[Finished]", end="")

        return response_content

    def parse_json_response(self, response: requests.Response):
        response_content = ""
        try:
            response_data = response.json()
            if self.api_format == "ollama":
                response_content = response_data["message"]["content"]
            else:
                response_content = response_data["choices"][0]["message"]["content"]
            logger.mesg(response_content)
            logger.success("[Finished]", end="")
        except Exception as e:
            logger.warn(f"× Error: {response.text}")
        return response_content

    def chat(
        self,
        messages: list,
        model: str,
        temperature: float = 0,
        seed: int = 42,
        stream=True,
    ):
        timer = Runtimer(verbose=False)
        timer.start_time()
        response = self.create_response(
            messages=messages,
            model=model,
            temperature=temperature,
            seed=seed,
            stream=stream,
        )
        if stream:
            response_content = self.parse_stream_response(response)
        else:
            response_content = self.parse_json_response(response)
        timer.end_time()
        elapsed_seconds = round(timer.elapsed_time().microseconds / 1e6, 1)
        logger.note(f" ({elapsed_seconds}s)")
        return response_content


if __name__ == "__main__":
    from configs.envs import LLMS_ENVS

    llm = LLMS_ENVS["deepseek"]
    args_dict = {
        "endpoint": llm["endpoint"],
        "api_key": llm["api_key"],
        "model": llm["model"],
        "api_format": llm["api_format"],
    }
    messages = [
        {
            "role": "system",
            "content": "你是一个由 Hansimov 开发的基于开源大语言模型搜索助手。你的任务是根据用户的输入，分析他们的意图和需求，生成搜索语句，调用搜索工具，最后提供用户所需的信息。",
        },
        {
            "role": "user",
            "content": "你是谁？",
        },
    ]
    client = LLMClient(endpoint, api_key, api_format)
    client.chat(messages, model, stream=True)

    # python -m llms.client
