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

"""
This example showcases recursive task decomposition to perform RAG which requires multiple steps.
The agent is a langchain custom LLM agent, which uses 2 tools - search and math.
It uses OpenAI's GPT-4 model for sub-answer formation, tool prediction and math operations. It uses the deployed LLM for final answer formation.
Search tool is a RAG pipeline, whereas the math tool uses an LLM call to perform mathematical calculations.
"""

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.agents import LLMSingleActionAgent, AgentOutputParser, AgentExecutor, Tool
from langchain.schema.agent import AgentFinish, AgentAction
from typing import List, Union
import json
import jinja2
import os

import base64
import os
import logging
from pathlib import Path
from typing import Generator, List

from llama_index import download_loader
from llama_index.node_parser import LangchainNodeParser

from RetrievalAugmentedGeneration.common.utils import (
    get_config,
    get_doc_retriever,
    get_llm,
    get_text_splitter,
    get_vector_index,
    is_base64_encoded,
    set_service_context,
)
from RetrievalAugmentedGeneration.common.base import BaseExample

logger = logging.getLogger(__name__)

FINAL_PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
Use the following context to answer the user's question. If you don't know the answer,
just say that you don't know, don't try to make up an answer.
<</SYS>>
<s>[INST] Context: {context} Question: {question} Only return the helpful
 answer below and nothing else. Helpful answer:[/INST]"
"""

llm = ChatOpenAI(model="gpt-4", temperature=0.1)
vector_db = None

##### Helper methods and tools #####

class Ledger:  # Stores the state of the recursive decomposition
    def __init__(self):
        self.question_trace = []
        self.answer_trace = []
        self.trace = 0
        self.done = False


##### LLM and Prompt definitions #####
def fetch_context(ledger: Ledger) -> str:
    """
    Create the context for the prompt from the subquestions and answers
    """
    context = ""
    for i in range(len(ledger.question_trace)):
        context += "Sub-Question: " + ledger.question_trace[i]
        context += "\nSub-Answer: " + ledger.answer_trace[i] + "\n"

    return context

template = """Your task is to answer questions. If you cannot answer the question, you can request use for a tool and break the question into specific sub questions. Fill with Nil where no action is required. You are given two tools:
- Search tool
- Math tool

Contextual Information:
{{ context }}

Question:
{{ question }}

{"Tool_Request": "<Fill>", "Generated Sub Questions": [<Fill>]}
"""

class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]
    ledger: Ledger

    def format_messages(self, **kwargs) -> str:
        kwargs["context"] = fetch_context(self.ledger).strip("\n")
        env = jinja2.Environment()
        prompt_template = env.from_string(template)
        prompt = prompt_template.render(**kwargs)
        logger.info(prompt)
        return [HumanMessage(content=prompt)]


##### LLM output parser #####


class CustomOutputParser(AgentOutputParser):
    class Config:
        arbitrary_types_allowed = True

    ledger: Ledger

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        Make a decision about the tool to be called based on LLM output.
        """

        logger.info(f"LLM Response: {llm_output}")
        local_state = json.loads(llm_output)
        if (
            local_state["Generated Sub Questions"][0] == "Nil"
            or local_state["Tool_Request"] == "Nil"
            or self.ledger.trace > 3
            or local_state["Generated Sub Questions"][0] in self.ledger.question_trace
        ):
            return AgentFinish(
                return_values={"output": "success"},
                log=llm_output,
            )

        if local_state["Tool_Request"] == "Search tool":
            self.ledger.trace += 1

        if local_state["Tool_Request"] in ["Search tool", "Math tool"]:
            return AgentAction(
                tool=local_state["Tool_Request"],
                tool_input={"sub_questions": local_state["Generated Sub Questions"]},
                log=llm_output,
            )
        raise ValueError(f"Invalid Tool name: {local_state['Tool_Request']}")


class QueryDecompositionChatbot(BaseExample):
    def ingest_docs(self, data_dir: str, filename: str):
        """Ingest documents to the VectorDB."""

        logger.info(f"Ingesting {filename} in vectorDB")
        _, ext = os.path.splitext(filename)

        if ext.lower() == ".pdf":
            PDFReader = download_loader("PDFReader")
            loader = PDFReader()
            documents = loader.load_data(file=Path(data_dir))

        else:
            unstruct_reader = download_loader("UnstructuredReader")
            loader = unstruct_reader()
            documents = loader.load_data(file=Path(data_dir), split_documents=False)

        encoded_filename = filename[:-4]
        if not is_base64_encoded(encoded_filename):
            encoded_filename = base64.b64encode(encoded_filename.encode("utf-8")).decode(
                "utf-8"
            )

        for document in documents:
            document.metadata = {"filename": encoded_filename}

        index = get_vector_index()
        node_parser = LangchainNodeParser(get_text_splitter())
        nodes = node_parser.get_nodes_from_documents(documents)
        index.insert_nodes(nodes)
        logger.info(f"Document {filename} ingested successfully")

    def llm_chain(self, context: str, question: str, num_tokens: int) -> Generator[str, None, None]:
        """Execute a simple LLM chain using the components defined above."""

        logger.info("Using llm to generate response directly without knowledge base.")
        set_service_context()
        prompt = get_config().prompts.chat_template.format(
            context_str=context, query_str=question
        )

        logger.info(f"Prompt used for response generation: {prompt}")
        response = get_llm().stream_complete(prompt, tokens=num_tokens)
        gen_response = (resp.delta for resp in response)
        return gen_response

    def rag_chain(self, question: str, num_tokens: int) -> Generator[str, None, None]:
        """Execute a Retrieval Augmented Generation chain using the components defined above."""

        logger.info("Using rag to generate response from document")

        set_service_context()
        if get_config().llm.model_engine == "triton-trt-llm":
            get_llm().llm.tokens = num_tokens  # type: ignore
        else:
            get_llm().llm.max_tokens = num_tokens

        global vector_db
        if not vector_db:
            vector_db = get_doc_retriever(num_nodes=4)

        final_context = self.run_agent(question)
        logger.info(f"Final Answer from agent: {final_context}")
        prompt = get_config().prompts.rag_template.format(
            context_str=final_context, query_str=question
        )

        logger.info(f"Prompt used for final response generation: {prompt}")
        response = get_llm().stream_complete(prompt, tokens=num_tokens)
        gen_response = (resp.delta for resp in response)

        return gen_response


    def create_agent(self) -> AgentExecutor:
        """
        Creates the tools, chain, output parser and agent used to fetch the full context.
        """

        self.ledger = Ledger()

        tools = [
            Tool(name="Search tool", func=self.search, description="Searches for the answer from a given context."),
            Tool(name="Math tool", func=self.math, description="Performs mathematical calculations."),
        ]
        tool_names = [tool.name for tool in tools]

        prompt = CustomPromptTemplate(template=template, tools=tools, input_variables=["question"], ledger=self.ledger)
        output_parser = CustomOutputParser(ledger=self.ledger)
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        recursive_decomposition_agent = LLMSingleActionAgent(
            llm_chain=llm_chain, output_parser=output_parser, stop=["\n\n"], allowed_tools=tool_names
        )

        agent_executor = AgentExecutor.from_agent_and_tools(agent=recursive_decomposition_agent, tools=tools, verbose=True)
        return agent_executor


    def run_agent(self, question: str):
        """
        Run question on the agent
        """

        agent_executor = self.create_agent()
        agent_executor.invoke({"question": question})

        ##### LLM call to get final answer ######

        prompt = "Question: " + question + "\n\n"
        prompt += "Sub Questions and Answers\n"
        for i in range(len(self.ledger.question_trace)):
            prompt += "Sub Question: " + str(self.ledger.question_trace[i]) + "\n"
            prompt += "Sub Answer: " + str(self.ledger.answer_trace[i]) + "\n"
        prompt += "\nFinal Answer: "

        return prompt

    def retriever(self, query: str) -> List[str]:
        """
        Searches for the answer from a given context.
        """

        result = vector_db.retrieve(query)
        logger.info(result)
        return [hit.node.text for hit in result]


    def extract_answer(self, chunks: List[str], question: str) -> str:
        """
        Find the answer to the query from the retrieved chunks
        """

        prompt = "Below is a Question and set of Passages that may or may not be relevant. Your task is to Extract the answer for question using only the information available in the passages. Do not infer or process the passage in any other way\n\n"
        prompt += "Question: " + question + "\n\n"
        for idx, chunk in enumerate(chunks):
            prompt += f"Passage {idx + 1}:\n"
            prompt += chunk + "\n"

        answer = llm([HumanMessage(content=prompt)])
        return answer.content


    def search(self, sub_questions: List[str]):
        """
        Search for the answer for each subquestion and add them to the ledger.
        """

        logger.info(f"Entering search with subquestions: {sub_questions}")
        for sub_question in sub_questions:
            chunk = self.retriever(sub_question)
            sub_answer = self.extract_answer(chunk, sub_question)

            self.ledger.question_trace.append(sub_question)
            self.ledger.answer_trace.append(sub_answer)


    def math(self, sub_questions: List[str]):
        """
        Places an LLM call to answer mathematical subquestions which do not require search
        """

        prompt = "Solve this mathematical question:\nQuestion: " + sub_questions[0]
        logger.info(f"Performing Math LLM call with prompt: {prompt}")
        sub_answer = llm([HumanMessage(content=prompt)])
        self.ledger.question_trace.append(sub_questions[0])
        self.ledger.answer_trace.append(sub_answer.content)

        self.ledger.done = True
