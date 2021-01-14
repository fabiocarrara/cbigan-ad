FROM tensorflow/tensorflow:2.4.0-gpu-jupyter

ADD requirements.txt /tmp/requirements.txt

RUN ["pip", "install", "-r", "/tmp/requirements.txt"]
