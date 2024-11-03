FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS build-py
LABEL authors="austin"
WORKDIR /home
COPY ./requirements.txt ./requirements.txt
<<<<<<< HEAD
RUN pip3 install -r requirements.txt
COPY ./get_cloud.py ./get_cloud.py
COPY ./config.py ./config.py
RUN python3 get_cloud.py

FROM build as run
=======
RUN pip3 install --root-user-action ignore -r requirements.txt

FROM build-py AS build-cloud
WORKDIR /home
COPY ./get_cloud.py ./get_cloud.py
RUN python3 get_cloud.py
RUN apt update
RUN apt install unzip
RUN unzip cloud_assets.zip -d cloud_assets
RUN rm cloud_assets.zip

FROM build-cloud AS run
>>>>>>> master
WORKDIR /home
COPY . .
ENTRYPOINT ["python3", "/home/main.py"]
#ENTRYPOINT ["bash"]
