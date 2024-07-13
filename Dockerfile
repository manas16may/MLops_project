FROM python:3.12
RUN apt-get update && apt-get install -y \
    libcoreclr \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python app.py