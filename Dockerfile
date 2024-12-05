FROM docker.io/pytorch/pytorch:latest AS build-1
LABEL authors="austin"
WORKDIR /tmp
COPY ./requirements.txt ./requirements.txt
RUN pip3 install --root-user-action ignore -r requirements.txt

FROM build-1 AS build
RUN apt update && apt install unzip
COPY ./get_cloud.py ./get_cloud.py
RUN python3 get_cloud.py \
    && unzip cloud_assets.zip \
    && rm cloud_assets.zip

FROM build AS copy
COPY ./text ./text
COPY ./image ./image
COPY ./like ./like
COPY ./*.py ./

ENTRYPOINT ["python3", "/tmp/main.py"]
