# install nvidia-docker
# https://github.com/NVIDIA/nvidia-docker
FROM tensorflow/tensorflow:latest-gpu

LABEL maintainer="Ujihiro Kazuya <kazuyaujihiro@gmail.com"

RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        bc \
        wget \
        vim \
        less \
        cmake ffmpeg qtbase5-dev python3-dev python-numpy python3-numpy libopencv-dev libgtk-3-dev libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev v4l-utils python-vtk liblapacke-dev libopenblas-dev checkinstall libgdal-dev \
        locales \
        python3-pip \
        python3-setuptools \
        && \
        rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        mock \
        numpy \
        scipy \
        sklearn \
        pandas \
        && \
    python -m ipykernel.kernelspec

RUN ln -s /usr/bin/python3 /usr/bin/python#

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN mkdir /work
WORKDIR "/work"

RUN python3 -m pip install --upgrade pip
# Install library
COPY requirements.txt ./
# RUN pip3 install -r requirements.txt
RUN python3 -m pip install -r requirements.txt

WORKDIR "/work"

RUN mkdir -p /home/src
RUN mkdir -p /home/res
RUN mkdir -p /home/originalres
WORKDIR "/home/src"

# Install TensorFlow GPU version.
# RUN pip3 install tensorflow-gpu
RUN python3 -m pip  install tensorflow-gpu

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

#WORKDIR "/notebooks"

#CMD ["/run_jupyter.sh"]
