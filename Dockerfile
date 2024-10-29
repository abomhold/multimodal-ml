FROM python:3.12.7-bookworm
LABEL authors="Masumi Yano"
LABEL name="py"
COPY . /home/
RUN pip3 install -r /home/requirements.txt

ENTRYPOINT ["bash"]