<!--
  SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# Structured Data

```{contents}
---
depth: 2
local: true
backlinks: none
---
```

## Example Features

This example deploys a developer RAG pipeline for chat Q&A and serves inferencing from an NVIDIA AI Foundation Models endpoint
instead of NVIDIA Triton Inference Server, a local Llama 2 model, or local GPUs.

Developers get free credits for 10K requests to any of the available models.

The key difference from [](./ai-foundation-models.md) example is that the example demonstrates how to use RAG with structured CSV data.

This example uses models from the NVIDIA AI Foundation Models and Endpoints.
This approach does not require embedding models or vector database solutions.
Instead, the example uses [PandasAI](https://docs.pandas-ai.com/en/latest/) to manage the workflow.

For ingestion, the query server loads the structured data from a CSV file into a Pandas dataframe.
The query server can ingest multiple CSV files, provided the files have identical columns.
Ingestion of CSV files with differing columns is not supported and results in an exception.

The core functionality uses a PandasAI agent to extract information from the dataframe.
This agent combines the query with the structure of the dataframe into an LLM prompt.
The LLM then generates Python code to extract the required information from the dataframe.
Subsequently, this generated code is executed on the dataframe and yields an output dataframe.

To demonstrate the example, sample CSV files are available.
These are part of the structured data example Helm chart and represent a subset of the [Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) from Kaggle.
The CSV data retrieval prompt is specifically tuned for three CSV files from this dataset: `PdM_machines.csv`, `PdM_errors.csv`, and `PdM_failures.csv`.
The CSV files to use are specified in the `rag-app-structured-data-chatbot.yaml` Docker Compose file by updating the environment variable `CSV_NAME`.
The default value is `PdM_machines`, but can be changed to `PdM_errors` or `PdM_failures`.
Customization of the CSV data retrieval prompt is not supported.

```{list-table}
:header-rows: 1

* - Model
  - Embedding
  - Framework
  - Description
  - Multi-GPU
  - TRT-LLM
  - NVIDIA AI Foundation
  - Triton
  - Vector Database

* - NV-Llama2-70B-RLHF
  - Not Applicable
  - PandasAI
  - QA chatbot
  - NO
  - NO
  - YES
  - NO
  - Not Applicable
```

The following figure shows the sample topology:

- The sample chat bot web application communicates with the chain server.
  The chain server sends inference requests to an NVIDIA AI Foundation Models endpoint.
- Optionally, you can deploy NVIDIA Riva. Riva can use automatic speech recognition to transcribe
  your questions and use text-to-speech to speak the answers aloud.

![Using NVIDIA AI Foundation Models endpoints for inference instead of local components.](./images/ai-foundations-topology.png)

## Prerequisites

- Clone the Generative AI examples Git repository using Git LFS:

  ```console
  $ sudo apt -y install git-lfs
  $ git clone git@github.com:NVIDIA/GenerativeAIExamples.git
  $ cd GenerativeAIExamples/
  $ git lfs pull
  ```

- Install Docker Engine and Docker Compose.
  Refer to the instructions for [Ubuntu](https://docs.docker.com/engine/install/ubuntu/).

- Optional: Enable NVIDIA Riva automatic speech recognition (ASR) and text to speech (TTS).

  - To launch a Riva server locally, refer to the [Riva Quick Start Guide](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html).

    - In the provided `config.sh` script, set `service_enabled_asr=true` and `service_enabled_tts=true`, and select the desired ASR and TTS languages by adding the appropriate language codes to `asr_language_code` and `tts_language_code`.

    - After the server is running, assign its IP address (or hostname) and port (50051 by default) to `RIVA_API_URI` in `deploy/compose/compose.env`.

  - Alternatively, you can use a hosted Riva API endpoint. You might need to obtain an API key and/or Function ID for access.

    In `deploy/compose/compose.env`, make the following assignments as necessary:

    ```bash
    export RIVA_API_URI="<riva-api-address/hostname>:<port>"
    export RIVA_API_KEY="<riva-api-key>"
    export RIVA_FUNCTION_ID="<riva-function-id>"
    ```

## Get an API Key for the Mixtral 8x7B Instruct API Endpoint

Perform the following steps if you do not already have an API key.
You can use different model API endpoints with the same API key.

1. Navigate to <https://catalog.ngc.nvidia.com/ai-foundation-models>.

1. Find the **Mixtral 8x7B Instruct** card and click **Learn More**.

   ![Mixtral 8x7B Instruct model card](./images/mixtral-8x7b-instruct.png)

1. Click the **API** button and then click **Generate Key**.

   ![API section of the playground tab.](./images/image8.png)

1. Save the generated API key.

## Build and Start the Containers

1. In the Generative AI examples repository, edit the `deploy/compose/compose.env` file.

   Add the API for the model endpoint:

   ```text
   ...
   # Fill this out if you dont have a GPU. Leave this empty if you have a local GPU
   export NVIDIA_API_KEY="nvapi-<...>"
   ...
   ```

1. From the root of the repository, build the containers:

   ```console
   $ docker compose --env-file deploy/compose/compose.env -f deploy/compose/rag-app-structured-data-chatbot.yaml build
   ```

1. Start the containers:

   ```console
   $ docker compose --env-file deploy/compose/compose.env -f deploy/compose/rag-app-structured-data-chatbot.yaml up -d
   ```

   *Example Output*

   ```output
    ✔ Network nvidia-rag         Created
    ✔ Container chain-server     Started
    ✔ Container rag-playground   Started
   ```

1. Confirm the containers are running:

   ```console
   $ docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
   ```

   *Example Output*

   ```output
   CONTAINER ID   NAMES               STATUS
   39a8524829da   rag-playground      Up 2 minutes
   bfbd0193dbd2   chain-server        Up 2 minutes
   ```

## Next Steps

- Access the web interface for the chat server.
  Refer to [](./using-sample-web-application.md) for information about using the web interface.
- Upload a CSV from the `RetrievalAugmentedGeneration/examples/csv_rag` directory to the knowledge base.
- Enable the **Use knowledge base** checkbox when you submit a question.
- [](./vector-database.md)
- Stop the containers by running `docker compose -f deploy/compose/rag-app-structured-data-chatbot.yaml down`.
