# Use an official Ubuntu runtime as a parent image
FROM ubuntu:16.04

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        imagemagick \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        python3 \
        python3-pip \
        python3-dev \
        python3-tk \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip3 --no-cache-dir install \
        Pillow \
        h5py \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        pyyaml \
        wheel \
        tensorflow

WORKDIR /main

# Copy the current directory contents into the container at /app
ADD . /main

# Run session.py when the container launches
CMD ["python3", "session.py"]