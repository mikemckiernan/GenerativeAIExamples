# Docker based RAG
Docker compose to manage enterprise RAG applications based on NVIDIA services.

# Examples supported
1. [Canonical RAG](#01-canonical-rag)


# 01: Canonical RAG

## Pre-requisites

## Description
This example showcases RAG pipeline. It uses nemollm inference microservice to host trt optimized llm and nemollm retriever embedding microservice. It uses pgvector as vectorstore to store embeddings and generate response for query.

<table class="tg">
<thead>
  <tr>
    <th class="tg-6ydv">LLM Model</th>
    <th class="tg-6ydv">Embedding</th>
    <th class="tg-6ydv">Framework</th>
    <th class="tg-6ydv">Document Type</th>
    <th class="tg-6ydv">Vector Database</th>
    <th class="tg-6ydv">Model deployment platform</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-knyo">llama-2-13b-chat</td>
    <td class="tg-knyo">nv-embed-qa</td>
    <td class="tg-knyo">llama-index</td>
    <td class="tg-knyo">PDF/Text</td>
    <td class="tg-knyo">PGVector</td>
    <td class="tg-knyo">On Prem</td>
  </tr>
</tbody>
</table>

## Deployment
1. Download llama2 model from NGC

    - Download the llama2 13b model from ngc
      ```
      ngc registry model download-version "ohlfw0olaadg/ea-participants/llama-2-13b-chat:LLAMA-2-13B-CHAT-4K-FP16-1-A100.24.01"
      ```

    - Move to the downloaded directory and unzip the model
      ```
      cd llama-2-13b-chat_vLLAMA-2-13B-CHAT-4K-FP16-1-A100.24.01/
      tar -xzf LLAMA-2-13B-CHAT-4K-FP16-1-A100.24.01.tar.gz
      ```

    - Check `model-store` directory after unzipping in the same directory.

2. Download the embedding model from ngc
    ```
    ngc registry model download-version "ohlfw0olaadg/ea-participants/nv-embed-qa:003"
    ```

3. Update model path in `compose.env` file with model environment variable
    - Update the absolute path of `model-store` directory of llama-2-chat model that you've downloaded and set it as `MODEL_DIRECTORY` environment variable  
      ```
export MODEL_DIRECTORY="/home/nvidia/llama2_13b_chat_hf_v1/model-store"
      ```
    
    - Update the absolute path of embedding model directory in environment variable `EMBEDDING_MODEL_DIRECTORY`

      ```
      export EMBEDDING_MODEL_DIRECTORY="/home/nvidia/nv-embed-qa_v003"
      ```

4. Run the pipeline
    ```
    source compose.env ; docker compose -f docker-compose-canonical-rag.yaml up -d
    ```

5. Check status of container
    ```
    $ docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
    CONTAINER ID   NAMES                                   STATUS
    32515fcb8ad2   rag-playground                          Up 26 minutes
    d60e0cee49f7   rag-application-text-chatbot            Up 27 minutes
    02c8062f15da   nemo-retriever-embedding-microservice   Up 27 minutes (healthy)
    7bd4d94dc7a7   nemollm-inference-ms                    Up 27 minutes
    4f191fbeda4a   pgvector                                Up 27 minutes
    ```

6. Open browser and interact with llm-playground at http://<host-ip>:8090