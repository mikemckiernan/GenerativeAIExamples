# NVIDIA Generative AI with AzureML Example

## Introduction
This example shows how to modify the conical RAG example to use a remote NVIDIA Nemotron-8B LLM hosted in AzureML. A custom LangChain connector is used to instantiate the LLM from within a sample notebook. To get started, deploy the conical RAG example and copy the 02.5_langchain_simple_AzureML.ipynb and trt_llm_azureml.py into the Jupyter environment. 
The Nemotron-8B models are curated by Microsoft in the ‘nvidia-ai’ Azure Machine Learning (AzureML)  registry and show up on the model catalog under the NVIDIA Collection. Explore the model card to learn more about the model architecture, use-cases and limitations.

## Large Language Models
NVIDIA LLMs are optimized for building enterprise generative AI applications.

| Name          | Description           | Type       | Context Length | Example | License |
|---------------|-----------------------|------------|----------------|---------|---------|
| [nemotron-3-8b-qa-4k](https://huggingface.co/nvidia/nemotron-3-8b-qa-4k) | Q&A LLM customized on knowledge bases | Text Generation | 4096 | No | [NVIDIA AI Foundation Models Community License Agreement](https://developer.nvidia.com/downloads/nv-ai-foundation-models-license) |
| [nemotron-3-8b-chat-4k-steerlm](https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm) | Best out-of-the-box chat model with flexible alignment at inference | Text Generation | 4096 | No | [NVIDIA AI Foundation Models Community License Agreement](https://developer.nvidia.com/downloads/nv-ai-foundation-models-license) |
| [nemotron-3-8b-chat-4k-rlhf](https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-rlhf) | Best out-of-the-box chat model performance| Text Generation | 4096 | No | [NVIDIA AI Foundation Models Community License Agreement](https://developer.nvidia.com/downloads/nv-ai-foundation-models-license) |
| [nemotron-3-8b-chat-sft](https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-sft) | building block for instruction tuning custom models, user-defined alignment, such as RLHF or SteerLM models. | Text Generation | 4096 | No | [NVIDIA AI Foundation Models Community License Agreement](https://developer.nvidia.com/downloads/nv-ai-foundation-models-license) |
| [nemotron-3-8b-base-4k](https://huggingface.co/nvidia/nemotron-3-8b-base-4k) | enables customization, including parameter-efficient fine-tuning and continuous pre-training for domain-adapted LLMs | Text Generation | 4096 | No | [NVIDIA AI Foundation Models Community License Agreement](https://developer.nvidia.com/downloads/nv-ai-foundation-models-license) |


## NVIDIA support
This example is experimental.

## Feedback / Contributions
We're posting these examples on GitHub to better support the community, facilitate feedback, as well as collect and implement contributions using GitHub Issues and pull requests. We welcome all contributions!

## Known issues
- In each of the READMEs, we indicate any known issues and encourage the community to provide feedback.
- The datasets provided as part of this project is under a different license for research and evaluation purposes.
- This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.