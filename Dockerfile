FROM python:2.7 AS deps

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "server.py"]