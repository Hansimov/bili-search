import asyncio
import json

from fastapi import WebSocket, WebSocketDisconnect
from tclogger import TCLogger
from functools import partial

from llms.agents.copilot import CopilotAgent


logger = TCLogger()


class WebsocketMessager:
    def parse(self, message: str):
        data = json.loads(message)
        message_type = data.get("type", None)
        message_info = data.get("info", None)
        return message_type, message_info


class WebsocketRouter:
    def __init__(self, ws: WebSocket, verbose: bool = False):
        self.ws = ws
        self.messager = WebsocketMessager()
        self.verbose = verbose

    async def run(self):
        await self.ws.accept()
        while True:
            msg_str = await self.ws.receive_text()
            try:
                msg_type, msg_info = self.messager.parse(msg_str)
                if msg_type == "chat":
                    copilot = CopilotAgent(
                        delta_func=self.ws.send_text, verbose_chat=False
                    )
                    await asyncio.to_thread(copilot.chat, msg_info)
            except WebSocketDisconnect:
                logger.success("* ws client disconnected")
                break
            except Exception as e:
                logger.warn(f"× ws error: {e}")
                break
