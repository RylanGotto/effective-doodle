import asyncio
import json
import logging
import os
import time
from asyncio import Lock, Semaphore
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from httpx import AsyncClient, HTTPStatusError, RequestError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("assistant.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Advanced Search Abstraction
class SearchProvider:
    """Abstract search provider with advanced error handling and logging."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.logger = logging.getLogger(self.__class__.__name__)

    async def news(self, query: str) -> List[Dict[str, str]]:
        """
        Perform a search with exponential backoff and retry mechanism.

        :param query: Search query string
        :return: List of search results
        """
        for attempt in range(self.max_retries):
            try:
                # Mock search implementation - replace with actual search logic
                results = [
                    {
                        "title": f"Result {i} for {query}",
                        "link": f"https://example.com/search?q={query}&page={i}",
                        "snippet": f"Sample search result {i} for {query}",
                    }
                    for i in range(3)
                ]
                return results
            except Exception as e:
                wait_time = 2**attempt  # Exponential backoff
                self.logger.warning(
                    f"Search attempt {attempt + 1} failed: {e}. Retrying in {wait_time} seconds."
                )
                await asyncio.sleep(wait_time)

        self.logger.error(f"Search failed after {self.max_retries} attempts")
        return []

    async def google(self, query: str) -> List[Dict[str, str]]:
        """
        Perform a search with exponential backoff and retry mechanism.

        :param query: Search query string
        :return: List of search results
        """
        for attempt in range(self.max_retries):
            try:
                # Mock search implementation - replace with actual search logic
                results = [
                    {
                        "title": f"Result {i} for {query}",
                        "link": f"https://example.com/search?q={query}&page={i}",
                        "snippet": f"Sample search result {i} for {query}",
                    }
                    for i in range(3)
                ]
                return results
            except Exception as e:
                wait_time = 2**attempt  # Exponential backoff
                self.logger.warning(
                    f"Search attempt {attempt + 1} failed: {e}. Retrying in {wait_time} seconds."
                )
                await asyncio.sleep(wait_time)

        self.logger.error(f"Search failed after {self.max_retries} attempts")
        return []


# Advanced Tools Management
class ToolsManager:
    """
    Comprehensive tools management with advanced features:
    - Centralized tool registration
    - Detailed logging
    - Error handling
    - Performance tracking
    """

    def __init__(self, search_provider: SearchProvider):
        self.search_provider = search_provider
        self._tools = {"news": self._news, "google": self._google}
        self._tool_calls = {}  # Track tool usage
        self._lock = Lock()

    async def invoke(self, tool_name: str, query: str) -> str:
        """
        Invoke a tool with comprehensive logging and performance tracking.

        :param tool_name: Name of the tool to invoke
        :param query: Query for the tool
        :return: Tool execution result
        """
        start_time = time.time()

        try:
            tool = self._tools.get(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")

            result = await tool(query)

            # Thread-safe tool call tracking
            async with self._lock:
                self._tool_calls[tool_name] = self._tool_calls.get(tool_name, 0) + 1

            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} invocation error: {e}")
            raise
        finally:
            duration = time.time() - start_time
            logger.info(f"Tool {tool_name} execution time: {duration:.4f} seconds")

    async def _news(self, query: str) -> str:
        """
        Perform web search with advanced error handling.

        :param query: Search query
        :return: JSON string of search results
        """
        try:
            results = await self.search_provider.news(query)
            return json.dumps(results[:3])
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return "[]"

    async def _google(self, query: str) -> str:
        """
        Perform web search with advanced error handling.

        :param query: Search query
        :return: JSON string of search results
        """
        try:
            results = await self.search_provider.news(query)
            return json.dumps(results[:3])
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return "[]"


class OpenAIAssistantManager:
    def __init__(
        self,
        tools_manager,
    ):
        _api_key = os.getenv("OPENAI_API_KEY")
        self.apy_key = _api_key
        self.model = os.getenv("OPENAI_MODEL")
        self.tools_manager = tools_manager

        self.headers = {
            "Authorization": f"Bearer {_api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2",
            "OpenAI-Organization": os.getenv("OPENAI_ORGANIZATION"),
            "OpenAI-Project": os.getenv("OPENAI_PROJECT"),
        }

        self.base_url = "https://api.openai.com/v1"

        self._assistant_id = os.getenv("OPENAI_ASSITANT_ID")
        self._thread_id = None

    async def _make_request(
        self, url: str, payload: Dict[str, Any], method: str = "POST"
    ) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if method.upper() == "POST":
                    response = await client.post(
                        url, json=payload, headers=self.headers
                    )
                else:
                    response = await client.get(url, headers=self.headers)

                response.raise_for_status()
                return response.json()
        except (RequestError, HTTPStatusError) as e:
            logger.error(f"Request error to {url}: {e}")
            raise

    async def send_message(self, message: str) -> AsyncGenerator[str, None]:
        """
        Send a message and stream the response with advanced processing.

        :param message: User message
        :return: Async generator of response tokens
        """
        # Create message in thread
        msg_response = await self.client.post(
            "{}/threads/{}/messages".format(self.base_url, self._thread_id),
            json={"role": "user", "content": message},
            headers=self.headers,
        )
        msg_response.raise_for_status()

        # Run thread
        run_response = await self.client.post(
            "{}/threads/{}/runs".format(self.base_url, self._thread_id),
            json={"assistant_id": self._assistant_id, "stream": True},
            headers=self.headers,
        )
        run_response.raise_for_status()

        # Process streaming response
        text = None
        state = {
            "index": 0,
            "function_calls": [],
            "arguments": [],
            "event": None,
            "run_id": 0,
        }

        async for line in run_response.aiter_lines():
            if line.startswith("event:"):
                state["event"] = line[7:]
            if line.startswith("data:"):
                try:
                    data = json.loads(line[5:])

                    text = await self._process_event_data(data, state)
                    if text:
                        yield text

                except json.JSONDecodeError:
                    continue

    async def _create_tool_outputs(self, tool_outputs):
        pass

    async def _create_tool_outputs(self, state):
        tool_outputs = []
        for i in state["function_calls"]:
            func = i.get("function")
            name = func.get("name")
            query = func.get("query")
            outputs = {
                "tool_call_id": i.get("id"),
                "output": await self.tools_manager.invoke(name, query),
            }
            tool_outputs.append(outputs)
        return self._submit_tools_and_run(self._submit_tools_and_run())

    async def _process_event_data(self, data: dict, state: dict) -> Optional[str]:
        """Process event data from the OpenAI response stream.

        Args:
            data (dict): Event data
            state (dict): Current state containing function_calls, arguments, index, event, run_id

        Returns:
            Optional[str]: Text response if available
        """

        if state["event"] == "thread.run.step.created":
            state["run_id"] = data.get("run_id", {})

        elif state["event"] == "thread.run.step.delta":
            tool_call = (
                data.get("delta", {}).get("step_details", {}).get("tool_calls", {})[0]
            )

            if "id" in tool_call:
                tool_call.update({"run_id": state["run_id"]})
                state["function_calls"].append(tool_call)
                state["index"] = tool_call.get("index", None)
                if state["index"] > 0:
                    state["function_calls"][state["index"] - 1].update(
                        {"arguments": json.loads("".join(state["arguments"]))}
                    )
                    state["arguments"].clear()
            else:
                state["arguments"].append(
                    tool_call.get("function", {}).get("arguments", "")
                )

        elif state["event"] == "thread.run.requires_action":
            state["function_calls"][state["index"]].get("function")["arguments"] = (
                json.loads("".join(state["arguments"]))
            )
            await self._submit_tools_and_run(state)

        else:
            content = data.get("delta", {}).get("content", [])
            if content:
                return content[0].get("text").get("value")

    @asynccontextmanager
    async def conversation_context(self):
        """
        Context manager for managing assistant conversation lifecycle.
        Handles thread creation, message processing, and cleanup.
        """
        self.client = httpx.AsyncClient()  # Create a single persistent client
        try:
            # Ensure assistant exists
            if not self._assistant_id:
                await self.create_assistant("Dynamic Assistant")

            # Create thread if needed
            if self._thread_id is None:
                thread_response = await self.client.post(
                    f"{self.base_url}/threads", headers=self.headers
                )
                thread_response.raise_for_status()
                self._thread_id = thread_response.json()["id"]
                logger.info(f"Conversation thread created: {self._thread_id}")

            yield self  # Yield self for use in the context

        except Exception as e:
            logger.error(f"Conversation context error: {e}")
            raise
        finally:
            await self.client.aclose()  # Properly close client when done
            self.client = None


# Main Execution
async def main():
    # Configuration from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "No OpenAI API key found. Set OPENAI_API_KEY environment variable."
        )
        return

    # Initialize components
    search_provider = SearchProvider()
    tools_manager = ToolsManager(search_provider)
    assistant_manager = OpenAIAssistantManager(tools_manager)

    # Interactive conversation loop
    try:
        async with assistant_manager.conversation_context() as assistant:
            while True:
                try:
                    user_input = input("\n>>> ")
                    if user_input.lower() in ["exit", "quit", "q"]:
                        break

                    async for response_chunk in assistant.send_message(user_input):
                        await asyncio.sleep(0.045)
                        print(response_chunk, end="", flush=True)

                except KeyboardInterrupt:
                    break
    except Exception as e:
        logger.error(f"Conversation error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
