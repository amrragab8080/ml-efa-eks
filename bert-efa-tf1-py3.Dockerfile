FROM amrragab8080/aws-efa-nccl-rdma:base-cudnn8-cuda11-ubuntu18.04

ARG TENSORFLOW_VERSION=1.15

###################################################
## Install Horovod and TensorFlow 1.x

# Install Tensorflow
RUN apt update && apt install -y cmake
RUN pip3 install -U pip
RUN pip3 install future typing mpi4py
RUN pip3 install numpy \
        tensorflow-gpu==${TENSORFLOW_VERSION} \
        keras \
        h5py


# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 pip3 install --no-cache-dir horovod && \
    ldconfig


###################################################
## Install Bert

RUN apt-get update && apt-get install -y pbzip2 pv bzip2 libcurl4 curl libb64-dev
RUN pip3 install --upgrade pip
RUN pip3 install toposort networkx pytest nltk tqdm html2text progressbar
RUN pip3 --no-cache-dir --no-cache install git+https://github.com/NVIDIA/dllogger

WORKDIR /workspace
RUN git clone https://github.com/openai/gradient-checkpointing.git
RUN git clone https://github.com/attardi/wikiextractor.git
RUN git clone https://github.com/soskek/bookcorpus.git
RUN git clone https://github.com/titipata/pubmed_parser


RUN pip3 install /workspace/pubmed_parser

#Copy the perf_client over
ARG TRTIS_CLIENTS_URL=https://github.com/NVIDIA/triton-inference-server/releases/download/v1.15.0/v1.15.0_ubuntu1804.clients.tar.gz
RUN mkdir -p /workspace/install \
    && curl -L ${TRTIS_CLIENTS_URL} | tar xvz -C /workspace/install

#Install the python wheel with pip
RUN pip install /workspace/install/python/tensorrtserver-1.15.0-py3-none-linux_x86_64.whl

WORKDIR /workspace/bert
COPY . .

ENV PYTHONPATH /workspace/bert
ENV BERT_PREP_WORKING_DIR /workspace/bert/data
ENV PATH //workspace/install/bin:${PATH}
ENV LD_LIBRARY_PATH /workspace/install/lib:${LD_LIBRARY_PATH}
RUN mkdir /results && touch /results/bert_logs.json
