from pydantic import BaseModel,Field
from typing import Optional, Union, List, Dict, Any, Literal
import json


class BaseMessage(BaseModel):
    content: Optional[str] = ""

    def dict(self) -> Dict:
        return dict(self)


class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"


class UserMessage(BaseMessage):
    role: Literal["user"] = "user"


class ToolMessage(BaseMessage):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: str


class AIMessage(BaseMessage):
    role: Literal["assistant"] = "assistant"
    tool_calls: Optional[List[Any]] = None


AnyMessage = Union[
    SystemMessage,
    UserMessage,
    AIMessage,
    ToolMessage,
]
