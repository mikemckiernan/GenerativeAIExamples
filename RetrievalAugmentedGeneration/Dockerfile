ARG BASE_IMAGE_URL=nvcr.io/nvidia/pytorch
ARG BASE_IMAGE_TAG=23.08-py3

FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG}
COPY RetrievalAugmentedGeneration/__init__.py /opt/RetrievalAugmentedGeneration/
COPY RetrievalAugmentedGeneration/common /opt/RetrievalAugmentedGeneration/common
COPY RetrievalAugmentedGeneration/examples /opt/RetrievalAugmentedGeneration/examples
COPY integrations /opt/integrations
RUN apt-get update && apt-get install -y libpq-dev
RUN pip3 install sqlalchemy[asyncio] pgvector psycopg2-binary asyncpg
RUN --mount=type=bind,source=RetrievalAugmentedGeneration/requirements.txt,target=/opt/requirements.txt \
    python3 -m pip install --no-cache-dir -r /opt/requirements.txt

WORKDIR /opt
ENTRYPOINT ["uvicorn", "RetrievalAugmentedGeneration.common.server:app"]
