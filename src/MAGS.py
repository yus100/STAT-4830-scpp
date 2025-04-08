from typing import List, Dict, Any, Tuple
from RAG import RAG
from config.config import Config
from LGK import LGK
from Graph import Graph, Node, Edge
from Memory import Memory, MemoryBlock, MemoryType, WorkingMemory, LTM, EpisodicMemory, SemanticMemory  
from Llama import LLama
from Embedding import Embedding
from HebPlasticity import HebPlasticity

import numpy as np

class MAGS:

    """
    Memory-Augmented Graphs System (Dynamic RAG + LKG)
    """

    def __init__(self):
        self.working_memory = LGK() # not used for now, will represent context in the future / nonpermanent information
        self.long_term_memory = LGK() # for now, a single graph for LTM
        self.llm = LLama(model_name="llama3.2")
        self.embeddor: Embedding = Embedding()
        self.config = Config()
        self.llm = LLama()
        self.heb_plasticity = HebPlasticity()

    async def add_text_to_LTM_Semantic(self, texts: list[str], node_adj: list[Node] = []):
        """
        Add texts to semantic long term memory, embed, create nodes, and initial connections
        """

        _embeded_text: list[list[np.ndarray]] = await self.embeddor.embed_documents(texts)
        
        _memoryblock: SemanticMemory = SemanticMemory(texts, _embeded_text)
        _node: Memory = Memory(MemoryType.SEMANTIC, _memoryblock)
        self.long_term_memory.add_node(_node)

        # create initial connections
        for _node_adj in node_adj:
            self.long_term_memory.add_edge(_node.id, _node_adj.id)
    

    async def update_memory(self, text_to_add: str, node: MemoryBlock):
        """
        Update memory with new text, embed, and update node
        """

        new_text = node._text + [ text_to_add ]
        _embeded_text: list[list[np.ndarray]] = await self.embeddor.embed_documents(new_text)

        await node.updatecontent(new_text, _embeded_text)
    
    async def query(self, question_prompt: str):
        documents = await self.recall_with_path_tracking(question_prompt)

        # refactored_questions = await RAG.multiquery(question_prompt, self.llm)

        # documents = await RAG.find_documents(refactored_questions, self.long_term_memory, top_k=3)

        documents_as_string = ""
        
        for index, document in enumerate(documents):
            documents_as_string += f"Document {index}: {document._memoryblock._text[0]}\n"

        prompt = self.config.get_prompt("rag_search")
        # replace memory and question WITHOUT formatting
        prompt = prompt.replace("{memory}", documents_as_string)
        prompt = prompt.replace("{question}", question_prompt)

        print(prompt)

        # 
        # .format(
        #     memory=documents_as_string,
        #     question=prompt
        # )

        response = (await self.llm.query_json(prompt))

        self.heb_plasticity.strengthen_paths(self.long_term_memory, reward=1.0)
        self.heb_plasticity.apply_recency_decay(self.long_term_memory)

        if "new_memory" in response["answer"]:
            if response["answer"]["new_memory"]["count"] > 0:
                count = response["answer"]["new_memory"]["count"]
                items = response["answer"]["new_memory"]["new_items"]

                for item in items:
                    await self.add_text_to_LTM_Semantic([item])
                
                print(f"ADDED {count} NEW MEMORIES")

        return response["answer"]["question_response"]

    async def recall_with_path_tracking(self, question_prompt: str, top_k: int = 3):
        # for hebbian updates -- path tracking
        refactored_questions = await RAG.multiquery(question_prompt, self.llm)
        documents = await RAG.find_documents(refactored_questions, self.long_term_memory, top_k=top_k)
        for doc in documents:
            # in a more sophisticated implementation, this would track the actual traversal path
            path = ["query_node", doc.id]
            self.heb_plasticity.record_path(path)
        
        return documents