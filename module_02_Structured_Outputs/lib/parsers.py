import json
from typing import Any, Type ,List
from abc import ABC, abstractmethod
from pydantic import BaseModel ,Field

from lib.messages import AIMessage


class OutputParser(BaseModel, ABC):
    @abstractmethod
    def parse(self, ai_message: AIMessage) -> Any:
        pass

class ActionItem(BaseModel):
    task: str = Field(..., description="Task description")
    assignee: str = Field(..., description="Person responsible for the task")
    due_date: str = Field(..., description="Due date for the task")



class StrOutputParser(OutputParser):
    def parse(self, ai_message: AIMessage) -> str:
        return ai_message.content


class ToolOutputParser(BaseModel):
    def parse(self, ai_message: AIMessage) -> list[dict]:
        return [{
            "tool_call_id":call.id,
            "args":json.loads(call.function.arguments),
            "function_name": call.function.name,
        } for call in ai_message.tool_calls]


class JsonOutputParser(OutputParser):
    def parse(self, ai_message: AIMessage) -> Any:
        return json.loads(ai_message.content)


class PydanticOutputParser(OutputParser):
    model_class: Type[BaseModel]

    def parse(self, ai_message: AIMessage) -> BaseModel:
        return self.model_class.model_validate_json(ai_message.content)

class MeetingSummary(BaseModel):
    title: str
    date: str
    participants: List[str]
    key_points: List[str]
    action_items: List[ActionItem]    