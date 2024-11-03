FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS build
LABEL authors="austin"
WORKDIR /home
COPY setup/requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
RUN apt update && apt install unzip
COPY setup/get_cloud.py ./get_cloud.py
RUN python3 get_cloud.py \
    && unzip -d cloud_assets/ cloud_assets.zip \
    && rm cloud_assets.zip

FROM build AS run
WORKDIR /home
COPY . .
ENTRYPOINT ["python3", "/home/main.py"]
#ENTRYPOINT ["bash"]
