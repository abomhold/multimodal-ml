FROM python:3.12.7-bookworm
LABEL authors="Masumi Yano"
LABEL name="py"
COPY . /home/
RUN pip3 install -r /home/requirements.txt
RUN python3 /home/preprocessing.py
ENTRYPOINT ["python3", "/home/main.py"]