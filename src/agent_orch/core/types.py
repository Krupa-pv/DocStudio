from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    role: Role
    content: str

    def to_openai_dict(self) -> Dict[str, Any]:
        #provide open ai with model context 
        return {"role": self.role.value, "content": self.content}


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    raw: Dict[str, Any]  
