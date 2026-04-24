from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import zlib

from dataclasses import asdict

from llms.models import DEFAULT_LARGE_MODEL_CONFIG, ModelRegistry, create_llm_client


DEFAULT_MODEL_CONFIG = DEFAULT_LARGE_MODEL_CONFIG


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


def _response_payload(response) -> dict:
    return {
        "content": response.content,
        "reasoning_content": response.reasoning_content,
        "usage": response.usage,
        "finish_reason": response.finish_reason,
    }


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
        "ok": "4" in content and "<think>" not in content.lower(),
        **_response_payload(response),
    }


def run_stream_text_check(client) -> dict:
    response = client.stream_to_response(
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
        "name": "stream_text",
        "ok": "4" in content and "<think>" not in content.lower(),
        **_response_payload(response),
    }


def run_xml_markup_check(client) -> dict:
    response = client.chat(
        messages=[
            {
                "role": "system",
                "content": "你是 XML 工具协议测试器。只输出一个自闭合 XML 工具标签，不要附加解释。",
            },
            {
                "role": "user",
                "content": "请输出一个 search_videos XML 标签，queries 只包含 黑神话。",
            },
        ],
        temperature=0,
    )
    content = str(response.content or "").strip()
    return {
        "name": "xml_markup",
        "ok": "<search_videos" in content
        and "黑神话" in content
        and "tool_calls" not in content.lower(),
        **_response_payload(response),
    }


def run_multimodal_check(client, image_url: str) -> dict:
    response = client.chat(
        messages=[
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
        ],
        temperature=0,
    )
    content = str(response.content or "").strip()
    lowered = content.lower()
    looks_reasonable = bool(content) and ("红" in content or "red" in lowered)
    return {
        "name": "multimodal",
        "ok": looks_reasonable,
        "image_url": image_url,
        **_response_payload(response),
    }


def _enabled_checks(spec, force_multimodal: bool) -> list[str]:
    checks = ["text", "stream_text", "xml_markup"]
    if force_multimodal or getattr(spec, "supports_multimodal", False):
        checks.append("multimodal")
    return checks


def main(default_model_config: str | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        default=default_model_config or DEFAULT_MODEL_CONFIG,
    )
    parser.add_argument("--image-url")
    parser.add_argument(
        "--force-multimodal",
        action="store_true",
        help="Run the multimodal probe even if the model registry marks it unsupported.",
    )
    args = parser.parse_args()

    registry = ModelRegistry.from_envs()
    spec = registry.get(args.model_config)
    if spec is None:
        raise SystemExit(f"Unknown model config: {args.model_config}")

    image_url = args.image_url or build_solid_color_png_data_url()
    client = create_llm_client(args.model_config, verbose=True)

    check_map = {
        "text": lambda: run_text_check(client),
        "stream_text": lambda: run_stream_text_check(client),
        "xml_markup": lambda: run_xml_markup_check(client),
        "multimodal": lambda: run_multimodal_check(client, image_url),
    }
    enabled_checks = _enabled_checks(spec, args.force_multimodal)
    checks = [check_map[name]() for name in enabled_checks]
    payload = {
        "model_config": args.model_config,
        "model_spec": asdict(spec),
        "enabled_checks": enabled_checks,
        "all_ok": all(item["ok"] for item in checks),
        "checks": checks,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
