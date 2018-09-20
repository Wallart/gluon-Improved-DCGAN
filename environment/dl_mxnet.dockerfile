ARG NVIDIA_STACK_VERSION=latest
FROM wallart/dl_nvidia:${NVIDIA_STACK_VERSION}
LABEL Author='Julien WALLART'

WORKDIR /tmp

ENV MXNET_VERSION 1.3.0
ENV OPENCV_VERSION 3.4.2

# Download frameworks
RUN git clone --recursive -b ${MXNET_VERSION} https://github.com/apache/incubator-mxnet mxnet-${MXNET_VERSION}
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz; mv ${OPENCV_VERSION}.tar.gz opencv-${OPENCV_VERSION}.tar.gz
RUN tar xf opencv-${OPENCV_VERSION}.tar.gz; rm -rf opencv-${OPENCV_VERSION}.tar.gz

# OpenCV deps
RUN apt update && apt install -y cmake ccache qtdeclarative5-dev libturbojpeg-dev libpng-dev libtiff-dev pkg-config

# Build OpenCV
RUN cd opencv-${OPENCV_VERSION}; mkdir build; cd build; cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D ENABLE_FAST_MATH=ON \
-D FORCE_VTK=OFF \
-D WITH_TBB=ON \
-D WITH_V4L=ON \
-D WITH_QT=ON \
-D WITH_OPENGL=ON \
-D WITH_GDAL=OFF \
-D WITH_XINE=ON \
-D WITH_MKL=ON \
-D MKL_ROOT_DIR=/opt/miniconda3/envs/intelmkl \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_opencv_dnn=OFF \
-D BUILD_opencv_legacy=OFF \
-D BUILD_opencv_python2=ON \
-D PYTHON2_EXECUTABLE=/opt/miniconda3/envs/intelpython2/bin/python \
-D PYTHON2_PACKAGES_PATH=/opt/miniconda3/envs/intelpython2/lib/python2.7/site-packages \
-D PYTHON2_LIBRARY=/opt/miniconda3/envs/intelpython2/lib/libpython2.7.so \
-D BUILD_opencv_python3=ON \
-D PYTHON3_EXECUTABLE=/opt/miniconda3/envs/intelpython3/bin/python \
-D PYTHON3_PACKAGES_PATH=/opt/miniconda3/envs/intelpython3/lib/python3.6/site-packages \
-D PYTHON3_LIBRARY=/opt/miniconda3/envs/intelpython3/lib/libpython3.6m.so \
-D PYTHON_DEFAULT_EXECUTABLE=/opt/miniconda3/envs/intelpython3/bin/python ..

RUN cd opencv-${OPENCV_VERSION}/build; make -j$(nproc); make install; rm -rf /tmp/opencv-${OPENCV_VERSION}

# MXNet deps
RUN apt install -y gcc-6 g++-6

# Symlink Intel MKL env to trick MXNet
RUN mkdir -p /opt/intel/mkl/lib
RUN ln -s /opt/miniconda3/envs/intelmkl/lib /opt/intel/mkl/lib/intel64

# Build MXNet
COPY config.mk mxnet-${MXNET_VERSION}/.
RUN cd mxnet-${MXNET_VERSION} && make -j$(nproc)

SHELL ["/bin/bash", "-c"]

# Install runtime dependencies
RUN /opt/miniconda3/bin/conda create -n intelmkl-dnn mkl-dnn
# Prepare env variables for all users
# Docker interactive mode
ENV LD_LIBRARY_PATH /opt/miniconda3/envs/intelmkl-dnn/lib:${LD_LIBRARY_PATH}
# For interactive login session
RUN echo "LD_LIBRARY_PATH=/opt/miniconda3/envs/intelmkl-dnn/lib:${LD_LIBRARY_PATH}" >> /etc/environment

# Install MXNet
RUN source /opt/miniconda3/bin/activate intelpython2; \
    pip install --upgrade pip; \
    pip uninstall --yes mxnet; \
    cd mxnet-${MXNET_VERSION}/python; \
    python setup.py install 2>&1 > /tmp/intelpython2-mxnet.log || echo 'Cannot downgrade numpy'
RUN source /opt/miniconda3/bin/activate intelpython3; \
    pip install --upgrade pip; \
    pip uninstall --yes mxnet; \
    cd mxnet-${MXNET_VERSION}/python; \
    python setup.py install 2>&1 > /tmp/intelpython3-mxnet.log || echo 'Cannot downgrade numpy'

# Runit startup
COPY bootstrap.sh /usr/sbin/bootstrap
RUN chmod 755 /usr/sbin/bootstrap

ENTRYPOINT ["/usr/sbin/bootstrap"]
