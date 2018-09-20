FROM ubuntu:17.10
LABEL Author='Julien WALLART'

   ################
  # System setup #
 ################

# Install mandatory packages for system configuration
RUN apt update && apt install -y passwd openssh-server make gcc g++ unzip net-tools dnsutils iproute2 openssh-client git wget python-pip vim

# Configure sshd
RUN ssh-keygen -A

# sshd ubuntu fix
RUN mkdir -p /run/sshd; cd /run/sshd; ln -s /etc/localtime localtime

  ###############
 # Runit setup #
###############

ENV RUNIT_VERSION=2.1.2
RUN mkdir -p /package
RUN chmod 1755 /package

ADD http://smarden.org/runit/runit-$RUNIT_VERSION.tar.gz runit-$RUNIT_VERSION.tar.gz
RUN tar xf runit-$RUNIT_VERSION.tar.gz
RUN cd admin/runit-$RUNIT_VERSION && package/install
RUN install -m0750 /admin/runit/etc/2 /usr/sbin/runsvdir-start
RUN mkdir -p /service
RUN rm -rf runit-$RUNIT_VERSION.tar.gz

# Adding services to runit
# SSH service
RUN mkdir -p /etc/service/sshd/
RUN echo '#!/bin/bash' > /etc/service/sshd/run
RUN echo '/usr/sbin/sshd -D -e' >> /etc/service/sshd/run
RUN chmod 755 /etc/service/sshd/run

  ################
 # Useful tools #
################

# Provide mungehosts
ADD https://github.com/hiteshjasani/nim-mungehosts/releases/download/v0.1.1/mungehosts /usr/local/bin/mungehosts
RUN chmod 755 /usr/local/bin/mungehosts

# Provide gosu
ADD https://github.com/tianon/gosu/releases/download/1.10/gosu-amd64 /usr/local/bin/gosu
RUN chmod 755 /usr/local/bin/gosu

# Runit startup
COPY bootstrap.sh /usr/sbin/bootstrap
RUN chmod 755 /usr/sbin/bootstrap

ENTRYPOINT ["/usr/sbin/bootstrap"]