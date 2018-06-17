#!/bin/bash -xe
#
##################################################################################
# outline
# run nlp-server(for training) docker
# $1:container id (e.g., 001)
# exit NORMAL:no error
# exit ABNORMAL END:-
##################################################################################
. kaggle.conf
docker run --name ${CONTAINER_NAME}-$1 \
  --log-driver=none \
  -it \
  -v /etc/localtime:/etc/localtime:ro \
  -v ${HOST_ROOT_DIRECTORY}/cpu/$1/src:/home/src \
  -v ${HOST_ROOT_DIRECTORY}/res:/home/res \
  ${CONTAINER_NAME}:${VERSION} /bin/bash
