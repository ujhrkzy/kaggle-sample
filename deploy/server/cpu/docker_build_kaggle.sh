#!/bin/bash -xe
#
##################################################################################
# outline
# build nlp-server(for training) docker
# $1:docker file path
# exit NORMAL:no error
# exit ABNORMAL END:-
##################################################################################
if [[ $# -ne 1 ]]; then
  echo "invalid arguments (expected:1, actual:$#)" 1>&2
  exit 1
fi

. kaggle.conf
docker build -t ${CONTAINER_NAME}:${VERSION} $1
