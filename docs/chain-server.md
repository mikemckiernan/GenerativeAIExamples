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

# Chain Server

```{contents}
---
depth: 2
local: true
backlinks: none
---
```

## About the Chain Server

The chain server is implemented as a sample FastAPI-based server so that you can experience a Q&A chat bot.
The server wraps calls made to different components and orchestrates the entire flow for all the generative AI examples.

## Endpoints

### Upload File Endpoint

**Summary:** Upload a file. This endpoint should accept a post request with the following JSON in the body:

```python
{
  "file": (file_path, file_binary_data, mime_type)
}
```

The response should be in JSON form. It should be a dictionary with a confirmation message:

```json
{"message": "File uploaded successfully"}
```

**Endpoint:** ``/uploadDocument``

**HTTP Method:** POST

**Request:**

- **Content-Type:** multipart/form-data
- **Required:** Yes

**Request Body Parameters:**
- ``file`` (Type: File) - The file to be uploaded.

**Responses:**
- **200 - Successful Response**

  - Description: The file was successfully uploaded.
  - Response Body: Empty

- **422 - Validation Error**

  - Description: There was a validation error with the request.
  - Response Body: Details of the validation error.



### Answer Generation Endpoint
**Summary:** Generate an answer to a question. This endpoint should accept a post request with the following JSON content in the body:

```json
{
  "question": "USER PROMPT",  // A string of the prompt provided by the user
  "context": "Conversation context to provide to the model.",
  "use_knowledge_base": false,  // A boolean flag to toggle VectorDB lookups
  "num_tokens": 500,  // The maximum number of tokens expected in the response.
}
```

The response should in JSON form. It should simply be a string of the response.

```json
"LLM response"
```

The chat server must also handle responses being retrieved in chunks as opposed to all at once. The client code for response streaming looks like this:

```python
with requests.post(url, stream=True, json=data, timeout=10) as req:
    for chunk in req.iter_content(16):
        yield chunk.decode("UTF-8")
```

**Endpoint:** ``/generate``

**HTTP Method:** POST

**Request:**

- **Content-Type:** application/json
- **Required:** Yes

**Request Body Parameters:**

-  ``question`` (Type: string) - The question you want to ask.
- ``context`` (Type: string) - Additional context for the question (optional).
- ``use_knowledge_base`` (Type: boolean, Default: true) - Whether to use a knowledge base.
- ``num_tokens`` (Type: integer, Default: 500) - The maximum number of tokens in the response.

**Responses:**

- **200 - Successful Response**

  - Description: The answer was successfully generated.
  - Response Body: An object containing the generated answer.

- **422 - Validation Error**

  - Description: There was a validation error with the request.
  - Response Body: Details of the validation error.

### Document Search Endpoint
**Summary:** Search for documents based on content. This endpoint should accept a post request with the following JSON content in the body:

```json
{
  "content": "USER PROMPT",  // A string of the prompt provided by the user
  "num_docs": "4",  // An integer indicating how many documents should be returned
}
```

The response should in JSON form. It should be a list of dictionaries containing the document score and content.

```json
[
  {
    "score": 0.89123,
    "content": "The content of the relevant chunks from the vector db.",
  },
  // ...
]
```


**Endpoint:** ``/documentSearch``
**HTTP Method:** POST

**Request:**

- **Content-Type:** application/json
- **Required:** Yes

**Request Body Parameters:**

- ``content`` (Type: string) - The content or keywords to search for within documents.
- ``num_docs`` (Type: integer, Default: 4) - The maximum number of documents to return in the response.

**Responses:**

- **200 - Successful Response**

  - Description: Documents matching the search criteria were found.
  - Response Body: An object containing the search results.

- **422 - Validation Error**

  - Description: There was a validation error with the request.
  - Response Body: Details of the validation error.

## API Reference

You can view the server API schema two ways:

- View it from ``http://host-ip:8081/docs``.
- View the [openapi_schema.json](./api_reference/openapi_schema.json) file.

## Running the Chain Server Independently

To run the server for development purposes, run the following commands:

- Build the container from source:

  ```console
  $ source deploy/compose/compose.env
  $ docker compose -f deploy/compose/rag-app-text-chatbot.yaml build chain-server
  ```

- Start the container, which starts the server:

  ```console
  $ source deploy/compose/compose.env
  $ docker compose -f deploy/compose/rag-app-text-chatbot.yaml up chain-server
  ```

- Open the swagger URL at ``http://host-ip:8081`` to try out the exposed endpoints.
