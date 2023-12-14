# Jupyter Notebooks
For development and experimentation purposes, the Jupyter notebooks provide guidance to building knowledge augmented chatbots.

The following Jupyter notebooks are provided with the AI enterprise workflow:

1. [**LLM Streaming Client**](../../notebooks/01-nemo-inference-ms-llm-streaming-client.ipynb)

This notebook demonstrates how to use a client to stream responses from an LLM deployed using [NeMo Microservice Inference (NMI)](https://registry.ngc.nvidia.com/orgs/ohlfw0olaadg/teams/ea-participants/containers/nemollm-inference-ms) which incorporates CUDA, TRT, TRT-LLM, and Triton, NMI brings state of the art GPU accelerated Large Language model serving.


# Running the notebooks
If a JupyterLab server needs to be compiled and stood up manually for development purposes, run the following commands:
- Build the container
```
source deploy/compose/compose.env
docker compose -f deploy/compose/docker-compose-enterprise.yaml build jupyter-server
```
- Run the container which starts the notebook server
```
source deploy/compose/compose.env
docker compose -f deploy/compose/docker-compose-enterprise.yaml up jupyter-server
```
- Using a web browser, type in the following URL to access the notebooks.

    ``http://host-ip:8888``
