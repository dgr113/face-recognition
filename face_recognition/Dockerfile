FROM python:3.7

ADD requirements.txt /opt/face_recognition/requirements.txt

WORKDIR /opt/face_recognition
RUN python3 -m pip install --no-cache-dir -r requirements.txt

ADD  project /opt/face_recognition/project

WORKDIR /opt/face_recognition/project
ENTRYPOINT ["python3", "cli.py"]
