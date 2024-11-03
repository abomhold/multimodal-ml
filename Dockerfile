FROM python:3.12.7-bookworm as build
LABEL authors="austin"
WORKDIR /home
#COPY ./requirements.txt ./requirements.txt
#RUN pip3 install -r requirements.txt
#COPY ./get_cloud.py ./get_cloud.py
#COPY ./config.py ./config.py
#RUN python3 get_cloud.py
#
#FROM build as run
#WORKDIR /home
COPY . .
ENTRYPOINT ["python3", "/home/main.py"]
#ENTRYPOINT ["bash"]
