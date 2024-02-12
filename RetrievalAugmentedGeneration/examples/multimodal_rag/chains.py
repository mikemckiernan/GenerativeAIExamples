# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Generator, List, Dict, Any

logger = logging.getLogger(__name__)

from RetrievalAugmentedGeneration.common.base import BaseExample
from RetrievalAugmentedGeneration.example.llm.llm_client import LLMClient
from RetrievalAugmentedGeneration.example.retriever.embedder import NVIDIAEmbedders
from RetrievalAugmentedGeneration.example.retriever.vector import MilvusVectorClient
from RetrievalAugmentedGeneration.example.retriever.retriever import Retriever
from RetrievalAugmentedGeneration.example.vectorstore.vectorstore_updater import update_vectorstore, load_documents, split_text


config = {
"header": "You are a helpful and friendly multimodal intelligent AI assistant named Multimodal Chatbot Assistant. You are an expert in the content of the document provided and can provide information using both text and images. The user may also provide an image input, and you will use the image description to retrieve similar images, tables and text. The context given below will provide some technical or financial documentation and whitepapers to help you answer the question. Based on this context, answer the question truthfully. If the question is not related to this, please refrain from answering. Most importantly, if the context provided does not include information about the question from the user, reply saying that you don't know. Do not utilize any information that is not provided in the documents below. All documents will be preceded by tags, for example [[DOCUMENT 1]], [[DOCUMENT 2]], and so on. You can reference them in your reply but without the brackets, so just say document 1 or 2. The question will be preceded by a [[QUESTION]] tag. Be succinct, clear, and helpful. Remember to describe everything in detail by using the knowledge provided, or reply that you don't know the answer. Do not fabricate any responses. Note that you have the ability to reference images, tables, and other multimodal elements when necessary. You can also refer to the image provided by the user, if any.",
"core_docs_directory_name": "multimodal"
}
sources = []
llm_client = LLMClient("mixtral_8x7b")

# init the retriever pipeline
try:
    vector_client = MilvusVectorClient(hostname="milvus", port="19530", collection_name=config["core_docs_directory_name"])
    query_embedder = NVIDIAEmbedders(name="nvolveqa_40k", type="query")
    document_embedder = NVIDIAEmbedders(name="nvolveqa_40k", type="passage")
    retriever = Retriever(embedder=query_embedder , vector_client=vector_client)
except Exception as e:
    logger.error(f"Failed to initialize the retriever pipeline with exception: {e}.")

logger.info(f"Successfully initialized multimodal rag pipeline.")


class MultimodalRAG(BaseExample):

    def ingest_docs(self, filepath: str, filename: str):
        """Ingest documents to the VectorDB."""

        try:
            update_vectorstore(os.path.abspath(filepath), vector_client, document_embedder, config["core_docs_directory_name"])
        except Exception as e:
            logger.error(f"Failed to ingest document due to exception {e}")
            raise ValueError("Failed to upload document. Please check chain server logs for details.")

    def llm_chain(
        self, context: str, question: str, num_tokens: str
    ) -> Generator[str, None, None]:
        """Execute a simple LLM chain using the components defined above."""

        logger.info("Using llm to generate response directly without knowledge base.")
        system_prompt = config["header"]
        response = llm_client.chat_with_prompt(system_prompt, question)
        return response

    def rag_chain(self, prompt: str, num_tokens: int) -> Generator[str, None, None]:
        """Execute a Retrieval Augmented Generation chain using the components defined above."""

        logger.info("Using rag to generate response from document")

        try:
            transformed_query = {"text": prompt}
            context, sources = retriever.get_relevant_docs(transformed_query["text"])
            augmented_prompt = "Relevant documents:" + context + "\n\n[[QUESTION]]\n\n" + transformed_query["text"]
            system_prompt = config["header"]
            response = llm_client.chat_with_prompt(system_prompt, augmented_prompt)
            return response

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

    def document_search(self, content: str, num_docs: int) -> List[Dict[str, Any]]:
        """Search for the most relevant documents for the given search parameters."""

        try:
            pass
        except Exception as e:
            logger.error(f"Error from /documentSearch endpoint. Error details: {e}")
        return []
