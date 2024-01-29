from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

class Embedder(ABC, BaseModel):
    

    @abstractmethod
    def embed_query(self, text):
        ...

    @abstractmethod
    def embed_documents(self, documents, batch_size):
        ...

    def get_embedding_size(self):
        sample_text = "This is a sample text."
        sample_embedding = self.embedder.embed_query(sample_text)
        return len(sample_embedding)

class NVIDIAEmbedders(Embedder):
    name : str
    type : str
    embedder : Optional[Any] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedder = NVIDIAEmbeddings(model=self.name, model_type=self.type)

    
    def embed_query(self, text):
        return self.embedder.embed_query(text)
    

    def embed_documents(self, documents, batch_size=10):
        output = []
        batch_documents = []
        for i, doc in enumerate(documents):
            batch_documents.append(doc)
            if len(batch_documents) == batch_size:
                output.extend(self.embedder.embed_documents(batch_documents))
                batch_documents = []
        else:
            if len(batch_documents) > 0:
                output.extend(self.embedder.embed_documents(batch_documents))
        return output


class HuggingFaceEmbeders(Embedder):
    name : str = "BAAI/bge-large-en-v1.5"
    embedder : Optional[Any] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedder = HuggingFaceEmbeddings(model_name=self.name)

    
    def embed_query(self, text):
        return self.embedder.embed_query(text)

    
    def embed_documents(self, documents, batch_size=5):
        output = []
        batch_documents = []
        for i, doc in enumerate(documents):
            batch_documents.append(doc)
            if len(batch_documents) == batch_size:
                output.extend(self.embedder.embed_documents(batch_documents))
                batch_documents = []
        else:
            if len(batch_documents) > 0:
                output.extend(self.embedder.embed_documents(batch_documents))
        return output