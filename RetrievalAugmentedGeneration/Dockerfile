ARG BASE_IMAGE_URL=nvcr.io/nvidia/pytorch
ARG BASE_IMAGE_TAG=23.11-py3


FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG}
COPY RetrievalAugmentedGeneration/__init__.py /opt/RetrievalAugmentedGeneration/
COPY RetrievalAugmentedGeneration/common /opt/RetrievalAugmentedGeneration/common
COPY RetrievalAugmentedGeneration/examples /opt/RetrievalAugmentedGeneration/examples
COPY integrations /opt/integrations
COPY RetrievalAugmentedGeneration/LICENSES-3rd-party.txt /opt
RUN --mount=type=bind,source=RetrievalAugmentedGeneration/requirements.txt,target=/opt/requirements.txt \
    python3 -m pip install --no-cache-dir -r /opt/requirements.txt; \
    apt-get clean

WORKDIR /opt
ENTRYPOINT ["uvicorn", "RetrievalAugmentedGeneration.common.server:app"]
