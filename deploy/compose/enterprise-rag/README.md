# Docker based RAG
Docker compose to manage enterprise RAG applications based on NVIDIA services.

# Examples supported
1. [Canonical RAG](#01-canonical-rag)
2. [No-GPU using NVIDIA AI Foundation](#02-no-gpu-using-nvidia-ai-foundation)
3. [Multi Modal RAG](#03-multi-modal-rag)
4. [MultiTurn RAG](#04-multi-turn-rag)


# 01: Canonical RAG

## Pre-requisites
1. Install [Docker Engine and Docker Compose.](https://docs.docker.com/engine/install/ubuntu/)

2. Verify NVIDIA GPU driver version 535 or later is installed.
    ```
    $ nvidia-smi --query-gpu=driver_version --format=csv,noheader
    535.129.03

    $ nvidia-smi -q -d compute

    ==============NVSMI LOG==============

    Timestamp                                 : Sun Nov 26 21:17:25 2023
    Driver Version                            : 535.129.03
    CUDA Version                              : 12.2

    Attached GPUs                             : 1
    GPU 00000000:CA:00.0
        Compute Mode                          : Default
    ```
    Reference: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and [NVIDIA Linux driver installation instructions](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)

3. Verify the NVIDIA container toolkit is installed and configured as the default container runtime.

    ```
    $ cat /etc/docker/daemon.json
    {
        "default-runtime": "nvidia",
        "runtimes": {
            "nvidia": {
                "path": "/usr/bin/nvidia-container-runtime",
                "runtimeArgs": []
            }
        }
    }

    $ sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi -L
    GPU 0: NVIDIA A100 80GB PCIe (UUID: GPU-d8ce95c1-12f7-3174-6395-e573163a2ace)
    ```

4. Create an NGC Account and API Key. Please refer to [instructions](https://docs.nvidia.com/ngc/gpu-cloud/ngc-overview/index.html) to create account and generate NGC API key.

    Login to `nvcr.io` using the following command:

    ```
    docker login nvcr.io
    ```

Reference:
- [Docker installation instructions (Ubuntu)](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit Installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

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

    - [Optional]If your model requires more than one GPU, export the `NUM_GPU` environment variable with the desired number of GPUs.
      ```
      export NUM_GPU=4 # Number of GPU required
      ```

4. Run the pipeline
    ```
    docker compose -f docker-compose-vectordb.yaml pgvector
    source compose.env ; docker compose -f rag-app-text-chatbot.yaml up -d
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

6. Open browser and interact with rag-playground at http://<host-ip>:8090


# 02: No-GPU using NVIDIA AI Foundation

## Pre-requisites
1. Install [Docker Engine and Docker Compose.](https://docs.docker.com/engine/install/ubuntu/)

2. Create an NGC Account and API Key. Please refer to [instructions](https://docs.nvidia.com/ngc/gpu-cloud/ngc-overview/index.html) to create account and generate NGC API key.

    Login to `nvcr.io` using the following command:

    ```
    docker login nvcr.io
    ```

Reference:
- [Docker installation instructions (Ubuntu)](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit Installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

3. Navigate to https://catalog.ngc.nvidia.com/ai-foundation-models

4. Find the <i>Mixtral x7B</i> model icon and click ``Learn More``.

5. Select the ```API``` navigation bar and click on the ```Generate key``` option.

6. Save the generated API key.

## Description
This example showcases a minimilastic RAG usecase built using Nvidia AI Foundation models which can answer questions from unstructured data based documents.

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
    <td class="tg-knyo">mixtral_8x7b</td>
    <td class="tg-knyo">nvolveqa_40k</td>
    <td class="tg-knyo">Langchain Expression Language</td>
    <td class="tg-knyo">PDF/Text</td>
    <td class="tg-knyo">PGVector</td>
    <td class="tg-knyo">Cloud - NVIDIA AI Foundation</td>
  </tr>
</tbody>
</table>

## Deployment
1. Add ngc api key in `compose.env`.
    ```shell
    export NVIDIA_API_KEY="nvapi-*"
    ```

2. Run the pipeline
    ```
    docker compose -f docker-compose-vectordb.yaml up -d pgvector
    source compose.env ; docker compose -f rag-app-ai-foundation-text-chatbot.yaml up -d
    ```

3. Check status of container
    ```
    $ docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
    CONTAINER ID   NAMES                                          STATUS
    32515fcb8ad2   rag-playground                                 Up 26 minutes
    d60e0cee49f7   rag-application-ai-foundation-text-chatbot     Up 27 minutes
    4f191fbeda4a   pgvector                                       Up 27 minutes
    ```

6. Open browser and interact with rag-playground at http://<host-ip>:8090

# 03: Multi Modal RAG

## Pre-requisites

Follow pre-requisites of [canonical rag](#pre-requisites) to install required dependency

## Description
This example showcases multi modal usecase in a RAG pipeline. It can understand any kind of images in PDF (like graphs and plots) alongside text and tables. It uses multimodal models from NVIDIA AI Foundation to answer queries.

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
    <td class="tg-knyo">mixtral_8x7b for response generation, Deplot for graph to text convertion, Neva 22B for image to text convertion</td>
    <td class="tg-knyo">nvolveqa_40k</td>
    <td class="tg-knyo">Custom Python</td>
    <td class="tg-knyo">PDF with images</td>
    <td class="tg-knyo">Milvus</td>
    <td class="tg-knyo">Cloud - NVIDIA AI Foundation</td>
  </tr>
</tbody>
</table>

## Deployment

1. Download mixtral_7x8b model from NGC

    - Download the mixtral-8x7b model from ngc
      ```
      ngc registry model download-version "ohlfw0olaadg/ea-participants/mixtral-8x7b:MIXTRAL-8x7b-INSTRUCT-2-A100.24.01"
      ```

    - Move to the downloaded directory and unzip the model
      ```
      cd mixtral-8x7b_vMIXTRAL-8x7b-INSTRUCT-2-A100.24.01/
      tar -xzf MIXTRAL-8x7b-INSTRUCT-2-A100.24.01.tar.gz
      ```

    - Check `model-store` directory after unzipping in the same directory.

2. Download the embedding model from ngc
    ```
    ngc registry model download-version "ohlfw0olaadg/ea-participants/nv-embed-qa:003"
    ```

3. Update model path in `compose.env` file with model environment variable
    - Update the absolute path of `model-store` directory of llama-2-chat model that you've downloaded and set it as `MODEL_DIRECTORY` environment variable
      ```
      export MODEL_DIRECTORY="/home/nvidia/mixtral/model-store"
      ```

    - Update the absolute path of embedding model directory in environment variable `EMBEDDING_MODEL_DIRECTORY`

      ```
      export EMBEDDING_MODEL_DIRECTORY="/home/nvidia/nv-embed-qa_v003"
      ```

    - Mixtral 8x7B model uses multiple gpu, update `EMBEDDING_MS_GPU_ID` with free GPU device id

      ```
      export EMBEDDING_MS_GPU_ID="2"
      ```

    - If your model requires more than one GPU, export the `NUM_GPU` environment variable with the desired number of GPUs. For mixtral-7x8b set it to 2
      ```
      export NUM_GPU=2 # Number of GPU required
      ```

4. Run the pipeline
    ```
    docker compose -f docker-compose-vectordb.yaml milvus

    source compose.env ; docker compose -f rag-app-multimodal-chatbot.yaml up -d
    ```

5. Check status of container
    ```
    $ docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
    CONTAINER ID   NAMES                                   STATUS
    32515fcb8ad2   rag-playground                          Up 26 minutes
    d60e0cee49f7   rag-application-multimodal-chatbot      Up 27 minutes
    02c8062f15da   nemo-retriever-embedding-microservice   Up 27 minutes (healthy)
    7bd4d94dc7a7   nemollm-inference-ms                    Up 27 minutes
    55135224e8fd   milvus-standalone                       Up 48 minutes (healthy)
    5844248a08df   milvus-minio                            Up 48 minutes (healthy)
    c42df344bb25   milvus-etcd                             Up 48 minutes (healthy)
    ```

6. Open browser and interact with rag-playground at http://<host-ip>:8090



# 04: Multi Turn RAG

## Pre-requisites

Follow pre-requisites of [canonical rag](#pre-requisites) to install required dependency

## Description
This example showcases multi turn usecase in a RAG pipeline. It stores the conversation history and knowledge base in PGVector and retrieves them at runtime to understand contextual queries. It uses NeMo Inference Microservices to communicate with the embedding model and large language model.
The example supports ingestion of PDF, .txt files. The docs are ingested in a dedicated document vectorstore. The prompt for the example is currently tuned to act as a document chat bot.
For maintaining the conversation history, we store the previous query of user and its generated answer as a text entry in a different dedicated vectorstore for conversation history.
Both these vectorstores are part of a Langchain [LCEL](https://python.langchain.com/docs/expression_language/) chain as Langchain Retrievers. When the chain is invoked with a query, its passed through both the retrievers.
The retriever retrieves context from the document vectorstore and the closest matching conversation history from conversation history vectorstore and the chunks are added into the LLM prompt as part of the chain.

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
    <td class="tg-knyo">Langchain Expression Language</td>
    <td class="tg-knyo">PDF/Text</td>
    <td class="tg-knyo">PGVector</td>
    <td class="tg-knyo">On Prem</td>
  </tr>
</tbody>
</table>

## Deployment

1. Download mixtral_7x8b model from NGC

    - Download the mixtral-8x7b model from ngc
      ```
      ngc registry model download-version "ohlfw0olaadg/ea-participants/mixtral-8x7b:MIXTRAL-8x7b-INSTRUCT-2-A100.24.01"
      ```

    - Move to the downloaded directory and unzip the model
      ```
      cd mixtral-8x7b_vMIXTRAL-8x7b-INSTRUCT-2-A100.24.01/
      tar -xzf MIXTRAL-8x7b-INSTRUCT-2-A100.24.01.tar.gz
      ```

    - Check `model-store` directory after unzipping in the same directory.

2. Download the embedding model from ngc
    ```
    ngc registry model download-version "ohlfw0olaadg/ea-participants/nv-embed-qa:003"
    ```

3. Update model path in `compose.env` file with model environment variable
    - Update the absolute path of `model-store` directory of llama-2-chat model that you've downloaded and set it as `MODEL_DIRECTORY` environment variable
      ```
      export MODEL_DIRECTORY="/home/nvidia/mixtral/model-store"
      ```

    - Update the absolute path of embedding model directory in environment variable `EMBEDDING_MODEL_DIRECTORY`

      ```
      export EMBEDDING_MODEL_DIRECTORY="/home/nvidia/nv-embed-qa_v003"
      ```

    - Mixtral 8x7B model uses multiple gpu, update `EMBEDDING_MS_GPU_ID` with free GPU device id

      ```
      export EMBEDDING_MS_GPU_ID="2"
      ```

    - If your model requires more than one GPU, export the `NUM_GPU` environment variable with the desired number of GPUs. For mixtral-7x8b set it to 2
      ```
      export NUM_GPU=2 # Number of GPU required
      ```

4. Run the pipeline
    ```
    docker compose -f docker-compose-vectordb.yaml pgvector

    source compose.env ; docker compose -f rag-app-multiturn-chatbot.yaml up -d
    ```

5. Check status of container
    ```
    $ docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
    CONTAINER ID   NAMES                                   STATUS
    32515fcb8ad2   rag-playground                          Up 26 minutes
    d60e0cee49f7   rag-application-multiturn-chatbot       Up 27 minutes
    02c8062f15da   nemo-retriever-embedding-microservice   Up 27 minutes (healthy)
    7bd4d94dc7a7   nemollm-inference-ms                    Up 27 minutes
    55135224e8fd   pgvector                                Up 27 minutes
    ```

6. Open browser and interact with rag-playground at http://<host-ip>:8090

