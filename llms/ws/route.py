import asyncio
import json

from collections import deque
from fastapi import WebSocket, WebSocketDisconnect
from tclogger import TCLogger
from typing import Literal, Any

from llms.agents.copilot import CopilotAgent


logger = TCLogger()

RECV_MESSAGE_TYPE = Literal["chat"]
SEND_MESSAGE_TYPE = Literal["chat", "tool"]


class WebsocketMessager:
    def parse(self, message: RECV_MESSAGE_TYPE):
        data = json.loads(message)
        message_type = data.get("type", None)
        message_info = data.get("info", None)
        return message_type, message_info


class WebsocketRouter:
    def __init__(self, ws: WebSocket, verbose: bool = False):
        self.ws = ws
        self.messager = WebsocketMessager()
        self.verbose = verbose
        self.terminate_event = asyncio.Event()
        self.disconnect_event = asyncio.Event()
        self.message_event = asyncio.Event()
        self.message_queue = deque()

    async def handle_recv_texts(self):
        while True:
            try:
                msg_str = await self.ws.receive_text()
                msg_type, msg_info = self.messager.parse(msg_str)
                if msg_type == "terminate":
                    self.terminate_event.set()
                    continue
                elif msg_type == "disconnect":
                    self.terminate_event.set()
                    self.disconnect_event.set()
                    await self.ws.close()
                    logger.success("* ws client disconnected", verbose=self.verbose)
                    break
                else:
                    self.message_queue.append((msg_type, msg_info))
                    self.message_event.set()
                    self.terminate_event.clear()
            except WebSocketDisconnect:
                logger.success("* ws client disconnected", verbose=self.verbose)
                break
            except Exception as e:
                logger.warn(f"× ws error: {e}")
                break
        self.terminate_event.set()
        self.disconnect_event.set()

    async def get_new_message(self) -> tuple[Literal["chat"], Any]:
        await self.message_event.wait()
        msg_type, msg_info = None, None
        if self.message_queue:
            msg_type, msg_info = self.message_queue.popleft()
        if not self.message_queue:
            self.message_event.clear()
        return msg_type, msg_info

    async def chat_delta_func(self, role: str, content: str, verbose: bool = False):
        msg = {"type": "chat", "info": {"role": role, "content": content}}
        await self.ws.send_text(json.dumps(msg))
        logger.mesg(content, end="", verbose=verbose)

    async def tool_func(self, info: list[dict], verbose: bool = False):
        msg = {"type": "tool", "info": info}
        await self.ws.send_text(json.dumps(msg))

    async def run(self):
        await self.ws.accept()
        asyncio.create_task(self.handle_recv_texts())
        while True:
            try:
                self.terminate_event.clear()
                msg_type, msg_info = await self.get_new_message()
                if msg_type == "chat":
                    copilot = CopilotAgent(
                        delta_func=self.chat_delta_func,
                        tool_func=self.tool_func,
                        terminate_event=self.terminate_event,
                        verbose_chat=False,
                    )
                    await asyncio.to_thread(copilot.chat, msg_info)
                    if self.terminate_event.is_set():
                        logger.success("* chat terminated", verbose=self.verbose)
                        continue
            except Exception as e:
                logger.warn(f"× run error: {e}")
                break
