FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS build-py
LABEL authors="austin"
WORKDIR /home
COPY ./requirements.txt ./requirements.txt
RUN pip3 install --root-user-action ignore -r requirements.txt

FROM build-py AS build-cloud
WORKDIR /home
COPY ./get_cloud.py ./get_cloud.py
RUN python3 get_cloud.py
RUN apt update && install unzip
RUN unzip cloud_assets.zip -d cloud_assets
RUN rm cloud_assets.zip

FROM build-cloud AS run
WORKDIR /home
COPY . .
ENTRYPOINT ["python3", "/home/main.py"]
#ENTRYPOINT ["bash"]
