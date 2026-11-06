"""NullProtocol: focused AI operations with managed context and observable results."""

from ._agent import Agent, AsyncAgent
from ._client import AsyncNullProtocol, NullProtocol
from ._errors import APIError, NullProtocolError, ProtocolError, TransportError
from ._types import ActionContext, Execution, KnowledgeUsage, Result, Usage

__version__ = "0.1.0"
__all__ = [
    "APIError",
    "ActionContext",
    "Agent",
    "AsyncAgent",
    "AsyncNullProtocol",
    "Execution",
    "KnowledgeUsage",
    "NullProtocol",
    "NullProtocolError",
    "ProtocolError",
    "Result",
    "TransportError",
    "Usage",
    "__version__",
]
