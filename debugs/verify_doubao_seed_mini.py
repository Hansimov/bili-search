from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import zlib

from llms.models import create_llm_client


DEFAULT_MODEL_CONFIG = "doubao-seed-2-0-mini"


def build_solid_color_png_data_url(
    *,
    width: int = 48,
    height: int = 48,
    rgb: tuple[int, int, int] = (255, 0, 0),
) -> str:
    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return (
            struct.pack("!I", len(data))
            + chunk_type
            + data
            + struct.pack("!I", checksum)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = png_chunk(
        b"IHDR",
        struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0),
    )
    row = bytes(rgb) * width
    raw = b"".join(b"\x00" + row for _ in range(height))
    idat = png_chunk(b"IDAT", zlib.compress(raw, level=9))
    iend = png_chunk(b"IEND", b"")
    payload = base64.b64encode(signature + ihdr + idat + iend).decode("ascii")
    return f"data:image/png;base64,{payload}"


def run_text_check(client) -> dict:
    response = client.chat(
        messages=[
            {
                "role": "user",
                "content": "请只回答一个阿拉伯数字：2+2 等于几？",
            }
        ],
        temperature=0,
    )
    content = str(response.content or "").strip()
    return {
        "name": "text",
        "ok": "4" in content,
        "content": content,
        "usage": response.usage,
        "finish_reason": response.finish_reason,
    }


def run_tool_check(client) -> dict:
    response = client.chat(
        messages=[
            {
                "role": "user",
                "content": "不要直接回答，必须调用 echo_city 工具，并把 city 设置为北京。",
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "echo_city",
                    "description": "回显用户指定的城市。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        temperature=0,
    )
    tool_calls = []
    for tool_call in response.tool_calls:
        tool_calls.append(
            {
                "name": tool_call.name,
                "arguments": tool_call.parse_arguments(),
            }
        )
    first_arguments = tool_calls[0]["arguments"] if tool_calls else {}
    return {
        "name": "tool_call",
        "ok": bool(tool_calls)
        and tool_calls[0]["name"] == "echo_city"
        and first_arguments.get("city") == "北京",
        "tool_calls": tool_calls,
        "usage": response.usage,
        "finish_reason": response.finish_reason,
        "content": response.content,
    }


def run_multimodal_check(client, image_url: str) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请只回答一个中文颜色词：这张图的主色调是什么？",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ]
    raw_response = client.create_response(
        messages=messages,
        model=None,
        enable_thinking=None,
        temperature=0,
        stream=False,
        extra_body=None,
    )
    if not raw_response.ok:
        error_body: object
        try:
            error_body = raw_response.json()
        except Exception:
            error_body = raw_response.text
        return {
            "name": "multimodal",
            "ok": False,
            "content": "",
            "usage": {},
            "finish_reason": "error",
            "image_url": image_url,
            "status_code": raw_response.status_code,
            "error_body": error_body,
        }
    response = client._parse_response(raw_response.json())
    content = str(response.content or "").strip()
    lowered = content.lower()
    looks_reasonable = bool(content) and ("红" in content or "red" in lowered)
    return {
        "name": "multimodal",
        "ok": looks_reasonable,
        "content": content,
        "usage": response.usage,
        "finish_reason": response.finish_reason,
        "image_url": image_url,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--image-url")
    args = parser.parse_args()

    image_url = args.image_url or build_solid_color_png_data_url()

    client = create_llm_client(args.model_config, verbose=True)
    checks = [
        run_text_check(client),
        run_tool_check(client),
        run_multimodal_check(client, image_url),
    ]
    payload = {
        "model_config": args.model_config,
        "all_ok": all(item["ok"] for item in checks),
        "checks": checks,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
