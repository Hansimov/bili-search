from sedb import LLMClientByConfig
from tclogger import JsonParser
from configs.envs import LLMS_ENVS
from llms.contants import LLM_CONFIG_TYPE


ENTITY_EXTRACT_PROMPT = """你是一个强大的命名实体分析引擎。请逐句分析，识别输入文本中所有的命名实体(类别：`人名`,`地名`,`专业术语`,`专有名词`)，并标记出最关键的命名实体（人名优先），输出JSON，用```包含，JSON 格式是 dict，输出样例是：
```json
{
    "entities": [
        {
            "entity": "...",
            "type": "..."
        },
        ...
    ],
    "core_entity": "..."
}
```
"""


class QueryEntityExtractor:
    def __init__(self, llm_config: LLM_CONFIG_TYPE = "qwen3-1.7b"):
        self.llm = LLMClientByConfig(LLMS_ENVS[llm_config])
        self.json_parser = JsonParser()

    def query_to_prompt(self, query: str) -> str:
        input_text = f"输入文本：\n```text\n{query}\n```\n"
        prompt = f"""{input_text}\n{ENTITY_EXTRACT_PROMPT}"""
        return prompt

    def get_entity(self, query: str) -> dict:
        resp_content = self.llm.chat(
            messages=[
                {
                    "role": "user",
                    "content": self.query_to_prompt(query),
                }
            ],
            temperature=0.6,
            enable_thinking=False,
            stream=False,
        )
        json_data = self.json_parser.parse(resp_content)
        if isinstance(json_data, dict):
            return json_data
        else:
            return {}
