#!/bin/bash

set -e

echo -n "本次更新镜像的版本(app) -> "
read APPTAG

sed -i "s/chatdata:v.*/chatdata:v$APPTAG/g" docker-compose-deploy.yml

docker pull xxx/chatdata:v${APPTAG}

docker-compose -f docker-compose-deploy.yml up -d
