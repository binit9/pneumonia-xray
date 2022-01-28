FROM bluedata/ubuntu18:latest

ENV LC_ALL "C.UTF-8"
ENV LANG "C.UTF-8"

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

EXPOSE 8501

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
# RUN pip install --upgrade streamlit
RUN pip install -r requirements.txt

ENTRYPOINT [ "streamlit", "run"]
CMD ["prediction_ui.py"]
