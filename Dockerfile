FROM python:3.12.7-bookworm
LABEL authors="austin"
COPY . /home
WORKDIR /home
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3", "/home/main.py"]
#ENTRYPOINT ["bash"]
