FROM python:latest

RUN mkdir -p /usr/src/submission-server
ADD ./*.py /usr/src/submission-server/
ADD ./requirements.txt /usr/src/submission-server

WORKDIR /usr/src/submission-server

RUN pip install numpy && \
    pip install Cython && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 5151

CMD ["python3", "server.py"]
