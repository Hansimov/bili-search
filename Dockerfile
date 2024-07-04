FROM ubuntu:jammy
WORKDIR /app
ARG UBUNTU_MIRROR
ARG PIP_MIRROR
ARG COMMIT_HASH
# Replace sources list
# https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/
RUN echo "deb $UBUNTU_MIRROR jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb $UBUNTU_MIRROR jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb $UBUNTU_MIRROR jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list
RUN apt-get update && apt-get install -y git python3.11 python3-pip
RUN git clone https://githubfast.com/Hansimov/bili-search.git . && git checkout $COMMIT_HASH
COPY requirements.txt ./
# https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
RUN pip3 install -i $PIP_MIRROR --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 21001
CMD ["python3.11", "-m", "apps.search_app"]
