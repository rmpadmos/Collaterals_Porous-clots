FROM ubuntu:18.04

# install requirements to setup `monodevelop` and `msbuild`
RUN apt-get update && apt-get install -y \
        apt-transport-https \
        software-properties-common \
        ca-certificates \
        dirmngr

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
RUN echo "deb https://download.mono-project.com/repo/ubuntu vs-bionic main" | tee /etc/apt/sources.list.d/mono-official-vs.list

# pull in more recent python versions
RUN add-apt-repository ppa:deadsnakes/ppa

# install `ubuntu` requirements
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        bash \
        monodevelop \
        msbuild \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-dev \
        python3.9 \
        python3.9-distutils \
        && rm -rf /var/lib/apt/lists/*

# compile 1-d unsteady model
WORKDIR /app/Pulsatile_Model/
COPY Pulsatile_Model .
RUN msbuild "Blood Flow Model.sln"
RUN chmod +x "Blood Flow Model/bin/Debug/BloodflowModel.exe"

WORKDIR /app

# install python dependencies
COPY in-silico-trial ./in-silico-trial
COPY requirements-container.txt ./
RUN pip3 install --upgrade cython
RUN pip3 install --no-cache-dir -r /app/requirements-container.txt

RUN python3.9 -m pip install pip --user
RUN python3.9 -m pip install --upgrade pip distlib wheel setuptools

# install the in-silico-trial package in 3.9
RUN cd /app/ && python3.9 -m pip install --no-cache-dir ./in-silico-trial

# copy all files
COPY . .

ENTRYPOINT ["python3.9", "/app/API.py"]
