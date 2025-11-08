import os
from agent_orch.core.config import Settings
from agent_orch.core.logging import setup_logging
from agent_orch.core.llm import LLMClient
from agent_orch.core.types import ChatMessage, Role


def main() -> None:
    #making logs readable
    cfg = Settings.load()
    setup_logging(cfg.log_level)

    #uncomment the next line if dont want API call to be made:
    # os.environ["DRY_RUN"] = "1"

    client = LLMClient(cfg)
    msgs = [
        ChatMessage(role=Role.SYSTEM, content="You are a concise assistant."),
        ChatMessage(role=Role.USER, content="Reply with the single word: apple"),
    ]
    result = client.chat(messages=msgs, temperature=0.0, max_tokens=5)

    print("TEXT:", result.text)
    print("TOKENS:", dict(prompt=result.prompt_tokens, completion=result.completion_tokens, total=result.total_tokens))
    print("LATENCY_MS:", round(result.latency_ms, 2))


if __name__ == "__main__":
    main()
