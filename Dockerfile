# Use an official Ubuntu runtime as a parent image
FROM osrf/ros:kinetic-desktop-xenial

# install ros packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    ros-kinetic-desktop-full=1.3.1-0* \
    build-essential \
    curl \
    imagemagick \
    libfreetype6-dev \
    libpng12-dev \
    libzmq3-dev \
    pkg-config \
    python \
    python-dev \
    # python3 \
    # python3-pip \
    # python3-dev \
    # python3-tk \
    rsync \
    software-properties-common \
    unzip \
    vim \
    wget

RUN apt-get remove python-numpy -y && \
    apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    # python3 get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# RUN pip3 --no-cache-dir install \
RUN pip --no-cache-dir install \
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


RUN rm /var/lib/dpkg/statoverride

RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
RUN wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add -
RUN apt-get update
RUN apt-get install --no-install-recommends -y \
	ros-kinetic-gazebo8-ros \
	ros-kinetic-gazebo8-ros-pkgs \
	ros-kinetic-gazebo8-ros-control
# RUN apt-get install ros-kinetic-gazebo-ros  --no-install-recommends -y \
# 		ros-kinetic-gazebo-ros-pkgs \
# 		ros-kinetic-gazebo-ros-control && \
# 	apt-get remove gazebo7 --no-install-recommends -y && \
# 	curl -ssL http://get.gazebosim.org | sh

RUN rosdep update && \
	echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc && \
	/bin/bash -c "source ~/.bashrc"

# Run session.py when the container launches
# CMD ["python3", "session.py"]
# CMD [ "gzserver", "--verbose &" ]