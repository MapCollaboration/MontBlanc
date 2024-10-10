FROM ubuntu:22.04

# Define shell and -c tells that the string that follow must be viewed as a command
SHELL ["/bin/bash", "-c"]


LABEL maintainer="Lorenzo Canzian"
LABEL description="Dockerfile for Fragmentation functions fit through MontBlanc"

ENV LANG=C.UTF-8
ENV LD_LIBRARY_PATH="/usr/local/lib/"

WORKDIR /home
RUN  mkdir lordguto
WORKDIR /home/lordguto

# Set version of all software needed for Montblanc and its dependecies
ARG NNAD_VERS=1.0
ARG NANGAPARBAT_VERS=1.5.0
ARG APFELXX_VERS=4.8.0
ARG CS_VERS=2.2.0
ARG LHAPDF_VERS=6.5.4
ARG ROOT_VERS=root_v6.30.06.Linux-ubuntu22.04-x86_64-gcc11.4

#update apt database
RUN echo deb http://dk.archive.ubuntu.com/ubuntu jammy main >> /etc/apt/sources.list \
 && echo deb http://dk.archive.ubuntu.com/ubuntu jammy-updates main >> /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y build-essential \
 && apt-get install -y ca-certificates

#install -y compilers
RUN apt-get install -y gfortran \
 && apt-get install -y g++ gcc  \
 && apt-get install -y python3 python3-dev python-is-python3

#install -y basic sysytem dependencies 
RUN apt-get install -y patch autoconf libtool make \
 && apt-get install -y gpg wget \
 && apt-get install -y vim \
 && apt-get install -y git

#install -y CMake
RUN wget https://apt.kitware.com/kitware-archive.sh \
 && chmod +x ./kitware-archive.sh && ./kitware-archive.sh \
 && apt-get install -y cmake \
 && rm kitware-archive.sh

##-- install -y project dependencies
#install -y yaml manager and numerical libraries
RUN apt-get install -y libyaml-cpp-dev \ 
 && apt-get install -y libeigen3-dev \
 && apt-get install -y libgsl-dev 

# Create directories for the dependencies

RUN mkdir Montblancinst
RUN mkdir NangaParbatinst
RUN mkdir NNADinst
RUN mkdir apfelxxinst
RUN mkdir lhapdf
RUN mkdir /home/lordguto/lhapdf/lhapdfinst
#RUN mkdir yamlinst

#install -y ceres-solver
RUN apt-get install -y libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev \
 && wget http://ceres-solver.org/ceres-solver-${CS_VERS}.tar.gz \
 && tar zxf ceres-solver-${CS_VERS}.tar.gz && rm ceres-solver-${CS_VERS}.tar.gz \
 && mkdir ceres-bin && cd ceres-bin \
 && cmake ../ceres-solver-${CS_VERS} && make && make test \
 && make install \
 && cd .. &&  rm -r ceres-solver-${CS_VERS} && rm -r ceres-bin 

#install -y ROOT
RUN apt-get install -y binutils dpkg-dev libssl-dev libx11-dev \libxext-dev libxft-dev libxpm-dev libtbb-dev \
 && wget https://root.cern/download/${ROOT_VERS}.tar.gz \ 
 && ln -sf /usr/share/zoneinfo/UTC /etc/localtime \
 && rm -rf /var/lib/apt/lists/*\
 && tar -xzvf ${ROOT_VERS}.tar.gz \
 && rm -f ${ROOT_VERS}.tar.gz \
 && echo home/lordguto/root/lib >> /etc/ld.so.conf \
 && ldconfig

 ENV ROOTSYS /home/lordguto/root \
     PATH $PATH:$ROOTSYS/bin \
     #PYTHONPATH $PYTHONPATH:$ROOTSYS/lib \
     CLING_STANDARD_PCH none

 #install -y lhapdf
ENV LHAPDF_DATA_PATH /home/lordguto/lhapdf/lhapdfinst/share/LHAPDF:$LHAPDF_DATA_PATH
ENV PYTHONPATH /home/lordguto/lhapdf/lhapdfinst/local/lib/python3.10/dist-packages

RUN cd /home/lordguto/lhapdf && wget https://lhapdf.hepforge.org/downloads/?f=LHAPDF-${LHAPDF_VERS}.tar.gz -O LHAPDF-${LHAPDF_VERS}.tar.gz \
 && tar -xf LHAPDF-${LHAPDF_VERS}.tar.gz && rm -r LHAPDF-${LHAPDF_VERS}.tar.gz \
 && cd LHAPDF-${LHAPDF_VERS} \
 && PYTHON=$(which python3) ./configure --prefix=/home/lordguto/lhapdf/lhapdfinst  && make && make install \
 && cd .. && rm -r LHAPDF-${LHAPDF_VERS}


ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/home/lordguto/lhapdf/lhapdfinst/lib
ENV PATH $PATH:/home/lordguto/lhapdf/lhapdfinst/bin

RUN lhapdf install NNPDF31_nnlo_pch_as_0118_1000 
RUN lhapdf install NNPDF31_nnlo_pch_as_0118
                  
#install -y apfel++
RUN cd /home/lordguto && wget https://github.com/vbertone/apfelxx/archive/refs/tags/${APFELXX_VERS}.tar.gz \
 && tar -xf ${APFELXX_VERS}.tar.gz && mv apfelxx-${APFELXX_VERS} apfelxx && rm -r ${APFELXX_VERS}.tar.gz \
 && cd apfelxx && mkdir build && cd build \ 
 && cmake -DCMAKE_INSTALL_PREFIX=/home/lordguto/apfelxxinst -DCMAKE_Fortran_COMPILER=/usr/bin/gfortran .. && make && make install \
 && echo /home/lordguto/apfelxxinst/lib >> /etc/ld.so.conf \
 && ldconfig

 #ENV LD_LIBRARY_PATH /home/lordguto/apfelxxinst/lib:$LD_LIBRARY_PATH
ENV PATH=/home/lordguto/apfelxxinst/bin:$PATH 
 
##-- install -y NangaParbat
RUN wget https://github.com/MapCollaboration/NangaParbat/archive/refs/tags/v${NANGAPARBAT_VERS}.tar.gz
RUN tar -xf v${NANGAPARBAT_VERS}.tar.gz && mv NangaParbat-${NANGAPARBAT_VERS} NangaParbat && rm -r v${NANGAPARBAT_VERS}.tar.gz
RUN source /home/lordguto/root/bin/thisroot.sh && cd NangaParbat && mkdir build && cd build \
 && cmake -DCMAKE_INSTALL_PREFIX=/home/lordguto/NangaParbatinst .. && make && make install

ENV LD_LIBRARY_PATH=/home/lordguto/NangaParbatinst/lib:$LD_LIBRARY_PATH
ENV PATH=/home/lordguto/NangaParbatinst/bin:$PATH 

##-- install -y NNAD
RUN wget https://github.com/rabah-khalek/NNAD/archive/refs/tags/v${NNAD_VERS}.tar.gz
RUN tar -xf v${NNAD_VERS}.tar.gz && mv NNAD-${NNAD_VERS} NNAD && rm -r v${NNAD_VERS}.tar.gz
RUN cd NNAD && mkdir build && cd build \
 && cmake -DCMAKE_INSTALL_PREFIX=/home/lordguto/NNADinst .. && make && make install

ENV LD_LIBRARY_PATH=/home/lordguto/NNADinst/lib:$LD_LIBRARY_PATH
ENV PATH=/home/lordguto/NNADinst/bin:$PATH 

# Add path and ld library path for ceres solver

ENV LD_LIBRARY_PATH=/usr/local/lib/cmake/Ceres/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/lib/cmake/Ceres/bin:$PATH 

##-- install -y MontBlanc
RUN git clone https://github.com/MapCollaboration/MontBlanc.git Montblanc
RUN cd Montblanc \
 && git checkout SIDIS \
 && mkdir build && cd build \
 && cmake -DCMAKE_INSTALL_PREFIX=/home/lordguto/Montblancinst .. && make && make install

ENV LD_LIBRARY_PATH=/home/lordguto/Montblancinst/lib:$LD_LIBRARY_PATH
ENV PATH=/home/lordguto/Montblancinst/bin:$PATH 

# Run Optimize
RUN cd Montblanc \
 && cd build \
 && cd run \
 && mkdir fit
 
ENTRYPOINT ["/bin/bash", "-c"]

CMD ["./Optimize", "1", "../../config/MAPFF20/PI/MAPFF20_PI_NNLO_Q1_00.yaml", "../../data/", "fit/"]


