import logging
import os
from functools import lru_cache
from typing import Generator

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from RetrievalAugmentedGeneration.common.base import BaseExample
from RetrievalAugmentedGeneration.common.utils import get_config

logger = logging.getLogger(__name__)
DOCS_DIR = os.path.abspath("./uploaded_files")
vector_store_path = "vectorstore.pkl"
document_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="passage")
vectorstore = None


@lru_cache
def get_llm() -> ChatNVIDIA:
    """Create the LLM connection."""
    llm = ChatNVIDIA(model="mixtral_8x7b")
    return llm


class NvidiaAIFoundation(BaseExample):
    def ingest_docs(self, file_name: str, filename: str):
        """Ingest documents to the VectorDB."""

        # TODO: Load embedding created in older conversation, memory persistance
        # We initialize class in every call therefore it should be global
        global vectorstore
        # Load raw documents from the directory
        # Data is copied to `DOCS_DIR` in common.server:upload_document
        settings = get_config()
        _path = os.path.join(DOCS_DIR, filename)
        raw_documents = UnstructuredFileLoader(_path).load()

        if raw_documents:
            # text_splitter = CharacterTextSplitter(chunk_size=settings.text_splitter.chunk_size, chunk_overlap=settings.text_splitter.chunk_overlap)
            text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            documents = text_splitter.split_documents(raw_documents)
            if vectorstore:
                vectorstore.add_documents(documents)
            else:
                vectorstore = FAISS.from_documents(documents, document_embedder)
            logger.info("Vector store created and saved.")
        else:
            logger.warning("No documents available to process!")

    def llm_chain(
        self, context: str, question: str, num_tokens: str
    ) -> Generator[str, None, None]:
        """Execute a simple LLM chain using the components defined above."""

        logger.info("Using llm to generate response directly without knowledge base.")
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are positive in nature.",
                ),
                ("user", "{input}"),
            ]
        )

        llm = get_llm()

        chain = prompt_template | llm | StrOutputParser()
        augmented_user_input = (
            "Context: " + context + "\n\nQuestion: " + question + "\n"
        )
        return chain.stream({"input": augmented_user_input})

    def rag_chain(self, prompt: str, num_tokens: int) -> Generator[str, None, None]:
        """Execute a Retrieval Augmented Generation chain using the components defined above."""

        logger.info("Using rag to generate response from document")

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant named Envie. You will reply to questions only based on the context that you are provided. If something is out of context, you will refrain from replying and politely decline to respond to the user.",
                ),
                ("user", "{input}"),
            ]
        )
        llm = get_llm()

        chain = prompt_template | llm | StrOutputParser()

        try:
            if vectorstore != None:
                retriever = vectorstore.as_retriever()
                docs = retriever.get_relevant_documents("List down member of twice")

                context = ""
                for doc in docs:
                    context += doc.page_content + "\n\n"

                augmented_user_input = (
                    "Context: " + context + "\n\nQuestion: " + prompt + "\n"
                )

                return chain.stream({"input": augmented_user_input})
        except Exception as e:
            logger.warning(f"Failed to generate response due to exception {e}")
        logger.warning(
            "No response generated from LLM, make sure you've ingested document."
        )
        return iter(
            [
                "No response generated from LLM, make sure you have ingested document from the Knowledge Base Tab."
            ]
        )
