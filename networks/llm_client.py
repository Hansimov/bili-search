import ast
import json
import re
import requests

from tclogger import logger, Runtimer


class LLMClient:
    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key

    def create_response(self, messages, model, stream=True):
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
            response_content = response_data["choices"][0]["message"]["content"]
            logger.mesg(response_content)
            logger.success("[Finished]", end="")
        except Exception as e:
            logger.warn(f"× Error: {response.text}")
        return response_content

    def chat(self, messages, model, stream=True):
        timer = Runtimer(verbose=False)
        timer.start_time()
        response = self.create_response(messages, model, stream)
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

    llm = LLMS_ENVS[0]
    endpoint = llm["endpoint"]
    api_key = llm["api_key"]
    model = llm["model"]
    messages = [
        {
            "role": "user",
            "content": "你是谁？",
        }
    ]
    client = LLMClient(endpoint, api_key)
    client.chat(messages, model, stream=True)

    # python -m networks.llm_client
