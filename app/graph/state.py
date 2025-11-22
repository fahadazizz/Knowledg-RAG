from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    steps: Annotated[List[str], operator.add]
