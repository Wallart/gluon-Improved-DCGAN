#!/bin/bash

# Add local user
# Either use the HOST_UID if passed in at runtime or
# fallback

USER_ID=${HOST_UID:-9001}
USER_NAME=${HOST_USER:-deeplearning}

echo "Starting as ${USER_ID}:${USER_NAME}"

# Create user if he does not exist
id -u $USER_NAME &> /dev/null
if [ $? -ne 0 ]; then
    # Without creating home dir because it could be a nfs mount
    useradd --shell /bin/bash -u $USER_ID -M -o -c "" $USER_NAME
    echo $USER_NAME:$USER_NAME | chpasswd

    # The first client of NFS will create the home dir
    if [ ! -d "/home/${USER_NAME}" ]; then
        mkdir -p /home/$USER_NAME
        echo "${USER_NAME} home directory ready"
    fi

    # And will populate with scripts from skel
    if [ -z $(find /home/${USER_NAME} -name ".bash*") ]; then
        cp /etc/skel/.bash* /home/$USER_NAME
        chown -R $USER_NAME:$USER_NAME /home/$USER_NAME
        echo "${USER_NAME} home directory populated"
    fi
fi

exec /usr/local/bin/runsvdir -P /etc/service
