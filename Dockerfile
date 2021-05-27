FROM python:2.7 AS deps

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 5000

CMD ["python", "server.py"]
