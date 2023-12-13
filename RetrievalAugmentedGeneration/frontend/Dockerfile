FROM nvcr.io/nvidia/pytorch:23.11-py3

COPY frontend /app/frontend
COPY requirements.txt /app
RUN apt-get update; \
    apt-get upgrade -y; \
    python3 -m pip --no-cache-dir install -r /app/requirements.txt; \
    apt-get clean
RUN apt-get remove python3-pip

USER 1001

WORKDIR /app
ENTRYPOINT ["python3", "-m", "frontend"]
