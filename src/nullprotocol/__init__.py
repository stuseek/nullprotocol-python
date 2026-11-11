"""NullProtocol: focused AI operations with managed context and observable results."""

from ._agent import Agent, AsyncAgent
from ._client import AsyncNullProtocol, NullProtocol
from ._errors import APIError, KnowledgeError, NullProtocolError, ProtocolError, TransportError
from ._knowledge import AgentKnowledge, AsyncAgentKnowledge, AsyncKnowledgeLibrary, KnowledgeLibrary
from ._types import ActionContext, Execution, KnowledgeUsage, Result, Usage

__version__ = "0.4.0"
__all__ = [
    "AgentKnowledge",
    "KnowledgeLibrary",
    "AsyncAgentKnowledge",
    "AsyncKnowledgeLibrary",
    "APIError",
    "ActionContext",
    "Agent",
    "AsyncAgent",
    "AsyncNullProtocol",
    "Execution",
    "KnowledgeUsage",
    "KnowledgeError",
    "NullProtocol",
    "NullProtocolError",
    "ProtocolError",
    "Result",
    "TransportError",
    "Usage",
    "__version__",
]
