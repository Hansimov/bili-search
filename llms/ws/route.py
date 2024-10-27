import asyncio
import json

from fastapi import WebSocket, WebSocketDisconnect
from functools import partial
from tclogger import TCLogger
from typing import Literal

from llms.agents.copilot import CopilotAgent


logger = TCLogger()

RECV_MESSAGE_TYPE = Literal["chat"]
SEND_MESSAGE_TYPE = Literal["chat"]


class WebsocketMessager:
    def parse(self, message: RECV_MESSAGE_TYPE):
        data = json.loads(message)
        message_type = data.get("type", None)
        message_info = data.get("info", None)
        return message_type, message_info

    def construct_chat_resp_dict(
        self, role: str, content: str, msg_type: SEND_MESSAGE_TYPE
    ):
        return {
            "type": msg_type,
            "info": {"role": role, "content": content},
        }


class WebsocketRouter:
    def __init__(self, ws: WebSocket, verbose: bool = False):
        self.ws = ws
        self.messager = WebsocketMessager()
        self.verbose = verbose

    async def chat_delta_func(self, role: str, content: str):
        msg = self.messager.construct_chat_resp_dict(role, content, msg_type="chat")
        await self.ws.send_text(json.dumps(msg))

    async def run(self):
        await self.ws.accept()
        while True:
            msg_str = await self.ws.receive_text()
            try:
                msg_type, msg_info = self.messager.parse(msg_str)
                if msg_type == "chat":
                    copilot = CopilotAgent(
                        delta_func=self.chat_delta_func, verbose_chat=False
                    )
                    await asyncio.to_thread(copilot.chat, msg_info)
            except WebSocketDisconnect:
                logger.success("* ws client disconnected")
                break
            except Exception as e:
                logger.warn(f"× ws error: {e}")
                break
