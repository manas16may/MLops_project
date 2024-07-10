FROM python:3.12

WORKDIR /app

COPY . /app

RUN pip install -r install.txt

CMD ["python3", "app.py"]