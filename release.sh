#!/bin/bash

set -e
echo -n "本次构建镜像的版本(app) -> "
read APPTAG
cat >docker-compose-build.yml <<EOF
version: '3.3'

services:
  app:
    container_name: chatdata
    image: chatdata:v${APPTAG}
    build:
      context: ./
      dockerfile: Dockerfile
    ports:
      - "8019:8019"
EOF
echo -e "生成build文件内容如下: "
echo -e "****************************************"
cat docker-compose-build.yml
echo -e "****************************************"
read -r -p "开始构建镜像，是否同步推送至Harbor [y/n]: " input
case ${input} in
[yY][eE][sS] | [yY])
  docker-compose -f docker-compose-build.yml build
  docker push xxx/chatdata:v${APPTAG}
  echo -e "Success ==> xxx/chatdata:v${APPTAG}"
  ;;
[nN][oO] | [nN])
  echo -e "开始构建本地镜像"
  docker-compose -f docker-compose-build.yml build
  echo -e "Success build local image!"
  ;;
*)
  echo "Invalid input..."
  ;;
esac

rm -f docker-compose-build.yml

# generate a new run file
cat >docker-compose-build.yml <<EOF
version: '3.3'

services:
  app:
    container_name: chatdata
    image: chatdata:v${APPTAG}
    build:
      context: ./
      dockerfile: Dockerfile
    ports:
      - "8019:8019"
EOF

echo -e "Complete!"
