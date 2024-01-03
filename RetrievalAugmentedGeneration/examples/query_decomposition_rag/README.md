# Query Decomposition Bot

## About

The query decomposition example showcases how to perform RAG when the agent needs to access information from several different files/chunks or perform some computation on the answers. It uses a custom langchain agent that recursively breaks down the user's questions into subquestions that it attempts to answer. It has access to 2 tools - search (which performs standard RAG on a subquestion) and math (which poses a math question to the LLM). The agent continues to break down the question into sub-questions until it has the answers it needs to formulate the final answer.

This agent uses the GPT-4 chat model from OpenAI for query decomposition, the search tool and the math tool. It uses the deployed LLM for generation of the final answer from the sub-questions and sub-answers.

An example where this agent is particularly useful is when dealing with multiple numerical figures. Let's say you have documents about NVIDIA's financial results for several years. Consider questions like:

- Which is greater - NVIDIA's datacenter revenue for Q3 2023 or its gaming revenue for Q1 2022?
- What is the sum of NVIDIA's datacenter and gaming revenue for Q3 2023 and Q4 2024?

The agent breaks down the question into sub-questions, like "What is NVIDIA's datacenter revenue for Q3 2023?" and "What is NVIDIA's gaming revenue for Q1 2022?", and individually answers these questions using RAG. FInally, the deployed model can combine the sub-answers into a final answer.

## Usage

1. Add your OPENAI API key to the compose.env file.
    ```shell
    export OPENAI_API_KEY=...
    ```

2. Change the name of the RAG example in compose.env.
    ```shell
    export RAG_EXAMPLE="query_decomposition_rag"
    ```

3. Add the OPENAI_API_KEY environment variable in the docker-compose file
    ```yaml
    environment:
      APP_MILVUS_URL: "http://milvus:19530"
      ...
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    ```

4. Start and interact with the chain server similar to other chains.