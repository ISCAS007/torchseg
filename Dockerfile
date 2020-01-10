###docker build -t youdaoyzbx/pytorch:torchseg .

FROM nvidia/cuda:9.0-runtime-ubuntu16.04
#ENV PATH /opt/conda/bin:$PATH
ENV PYTHONPATH .
WORKDIR /workspace
COPY requirements.txt .
COPY conda.yml .
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 vim \
    git mercurial subversion
#    build-essential gcc

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
# RUN conda create -n torch1.0 --file requirements.txt --channel conda-forge --channel pytorch
# RUN conda env update -f conda.yml -n base
# RUN conda create -n torch1.0 && conda activate torch1.0 && \
#    conda install pip && pip install -r requirements.txt # fail with eigen version
RUN conda create -n torch1.0 -f conda.yml # avoid change base