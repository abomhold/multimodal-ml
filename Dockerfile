FROM python:3.12.7-bookworm
LABEL authors="austin"
RUN pip3 install -r /home/requirements.txt
ENTRYPOINT ["python3", "main.py"]