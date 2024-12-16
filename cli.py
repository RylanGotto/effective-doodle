import asyncio

from OpenAIAssitantManager import _start


async def main():
    # Configuration from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "No OpenAI API key found. Set OPENAI_API_KEY environment variable."
        )
        return

    # Initialize components
    assistant_manager = _start()

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
        print(f"Conversation error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
