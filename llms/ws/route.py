import json

from fastapi import WebSocket

from llms.agents.copilot import CopilotAgent


class WebsocketMessager:
    def parse(self, message: str):
        data = json.loads(message)
        message_type = data.get("type", None)
        message_info = data.get("info", None)
        return message_type, message_info


class WebsocketRouter:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.messager = WebsocketMessager()

    async def run(self):
        await self.ws.accept()
        while True:
            msg_str = await self.ws.receive_text()
            try:
                msg_type, msg_info = self.messager.parse(msg_str)
                if msg_type == "chat":
                    copilot = CopilotAgent()
                    response = await copilot.chat(msg_info)
                await self.ws.send_text(response)
            except:
                pass
