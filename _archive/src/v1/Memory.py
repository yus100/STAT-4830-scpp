from Graph import Node
from enum import Enum

import numpy as np

class MemoryType(Enum):
    UNKNOWN = 0
    WORKING = 1
    LTM = 2
    EPISODIC = 3
    SEMANTIC = 4


class MemoryBlock():
    def __init__(self, text: list[str], embeddings: list[list[np.ndarray]]):
        pass
        self.activation_count = 0
        self.last_activation = 0

        self._emeddings: list[list[np.ndarray]] = embeddings
        self._text: list[str] = text

    async def activate(self, activation_time: int):
        self.activation_count += 1
        self.last_activation = activation_time
    
    async def updatecontent(self, text: list[str], embeddings: list[list[np.ndarray]]):
        self._emeddings = embeddings
        self._text = text

    async def get_embeddings(self):
        return self._emeddings
    async def get_text(self):
        return self._text


class Memory(Node):
    def __init__(self, type: MemoryType = MemoryType.UNKNOWN, memoryblock: MemoryBlock | None = None):
        super().__init__()
        self._memorytype: MemoryType = type
        self._memoryblock : MemoryBlock | None = memoryblock

    def __repr__(self):
        return f"Memory({self.id}, {self._memorytype}, {self._memoryblock})"


class WorkingMemory(MemoryBlock):
    """
    Used for storing context and nonpermanent information for a conversation.
    """
    pass

class LTM(MemoryBlock):
    """
    Used for storing facts and information that should be long-term.
    """
    pass

class EpisodicMemory(LTM):
    """
    Storing events and experiences
    """
    pass

class SemanticMemory(LTM):
    """
    Storing facts
    """
    def __init__(self, text: list[str], embedding: list[list[np.ndarray]]):
        super().__init__(text, embedding)
        

