import asyncio
import json
import time
from abc import ABC, abstractmethod
from pprint import pprint
from typing import *

import httpx

from search import Search
from tools import BaseTools, WebSearch


class OpenAIAssistancesV2Stream:
    def __init__(
        self,
        model: str,
        api_key: str,
        name: str,
        desc: Optional[str],
        tools: BaseTools,
        asst_id: Optional[str] = None,
    ):
        self.tools = tools
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2",
            "OpenAI-Organization": "org-d8UZEJoq8jw80KOsCEVLy354",
            "OpenAI-Project": "proj_LWkbTxZGMJZV2gVG5wZsDIGz",
        }
        self.model = model
        self.create_url = "https://api.openai.com/v1/assistants"
        self.thread_url = "https://api.openai.com/v1/threads"
        self.assistant_id = asst_id
        self.thread_id = None
        self.create_message_url = None
        self.run_thread_url = None
        self.run_id = None
        self.return_value = None
        if self.assistant_id is None:
            self.assistant_id = self.create_assistant(name, desc)

    async def make_request(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()

    async def delete_assistant(self) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(self.create_url, headers=self.headers)
            response.raise_for_status()
            for assistant in response.json()["data"]:
                await client.delete(
                    f"{self.create_url}/{assistant['id']}", headers=self.headers
                )

    async def create_assistant(self, name: str, desc: Optional[str]) -> str:
        payload = {
            "name": name,
            "instructions": desc,
            "model": self.model,
        }
        response = await self.make_request(self.create_url, payload)
        return response["id"]

    async def create_thread(self) -> str:
        response = await self.make_request(self.thread_url, {})

        self.create_message_url = f"{self.thread_url}/{response['id']}/messages"
        self.run_thread_url = f"{self.thread_url}/{response['id']}/runs"
        return response["id"]

    async def create_message(self, message: str) -> Dict[str, Any]:
        payload = {
            "role": "user",
            "content": message,
        }

        return await self.make_request(self.create_message_url, payload)

    async def run_thread(self) -> None:
        payload = {"assistant_id": self.assistant_id, "stream": True}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.run_thread_url, json=payload, headers=self.headers
            )
            return response

    async def process_response(self, response):
        tool_calls = []
        tool_call_arguments = []
        arguments = []
        run_id = {}
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            print(line)
            data = line[6:]
            if data == "[DONE]":
                merged_list = [
                    {**tool_call, **tool_call_argument, **run_id}
                    for tool_call, tool_call_argument in zip(
                        tool_calls, tool_call_arguments
                    )
                ]
                if merged_list:
                    r = await self.submit_tool_output_to_run(merged_list)
                    # Instead of yielding the nested generator, yield its flattened content.
                    async for text in self.process_response(
                        r
                    ):  # Flattening the nested generator
                        yield text
                break

            data = json.loads(data)
            if data.get("run_id", {}):
                self.run_id = data.get("run_id", {})
            delta = data.get("delta", {})
            step_details = delta.get("step_details", {})
            content = delta.get("content", [])
            if step_details:
                tool_call = step_details.get("tool_calls", [{}])[0]
                if "id" in tool_call:
                    arguments.clear()
                    tool_calls.append(tool_call)
                else:
                    arguments.append(tool_call.get("function", {}).get("arguments", ""))
                    try:
                        tool_call_arguments.append(json.loads("".join(arguments)))
                    except json.JSONDecodeError:
                        continue

            if content:
                time.sleep(0.05)
                text = content[0].get("text", {}).get("value", "")
                x = json.dumps({"choices": [{"delta": {"content": text}}]})
                data = f"data: {x}\n\n"
                yield data
                # print(text, end="", flush=True)

    async def submit_tool_output_to_run(self, tool_calls):
        tasks = []

        for i in tool_calls:
            tasks.append(self.create_function(i, self.tools))

        results = await asyncio.gather(*tasks)
        payload = {"tool_outputs": results, "stream": True}
        url = f"{self.run_thread_url}/{self.run_id}/submit_tool_outputs"
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.post(url, json=payload, headers=self.headers)
            return response

    async def create_function(self, func_call, tools):
        call_id = func_call.get("id")
        func_name = func_call.get("function").get("name")
        query = func_call.get("query")
        result = await tools.invoke(func_name, query)
        return {"tool_call_id": call_id, "output": str(result)}

    async def return_dict(self, call_id, func):
        return {"tool_call_id": call_id, "output": func}

    async def send(self, message: str) -> None:
        await self.create_message(message)
        response = await self.run_thread()
        return self.process_response(response)


async def main():
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

    prompt = input(">>> ")
    x = await assistant.send(prompt)
    async for i in x:
        print(i, end="", flush=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
