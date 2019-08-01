FROM python:3.7

WORKDIR /app

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --upgrade --no-cache-dir -r /tmp/requirements.txt

COPY ./ckpt /app/ckpt
COPY ./identification /app/identification
COPY ./recognition /app/recognition
COPY ./vectorizer /app/vectorizer

CMD [ "python", "-m", "vectorizer.server" ]
