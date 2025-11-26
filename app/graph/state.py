from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    # We can keep other keys if needed, but 'messages' is the core for chat
    documents: List[str] # To store retrieved context explicitly if needed
