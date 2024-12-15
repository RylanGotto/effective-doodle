import json
from typing import *

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from search import Search
from stream2 import OpenAIAssistancesV2Stream
from tools import WebSearch

app = FastAPI()


async def generate_response(prompt):
    api_key = ""
    name = "Personal Assistant Helper"
    assistant = OpenAIAssistancesV2Stream(
        model="gpt-3.5-turbo-0125",
        api_key=api_key,
        name=name,
        desc=None,
        asst_id="asst_WQiJkKKXOs4mSC1g4lJiroxQ",
        tools=WebSearch(Search()),
    )
    await assistant.create_thread()
    x = await assistant.send(prompt)
    return x


@app.post("/chat/")
async def chat(request: Request):
    x = await request.body()
    x = json.loads(x)
    print()
    y = await generate_response(x.get("messages").pop().get("content"))
    return StreamingResponse(
        y,
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
