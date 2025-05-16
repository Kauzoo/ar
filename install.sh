#!/bin/sh
EXTENSION_API_PATH=../gdextension-api-files/gd4.4_linux_extension_api.json
PLATFORM=linux
EXPORT_DIR=./opencv
printf "Running default linux install"
printf "REQUIREMENTS"
printf "Docker (installed and running)\nScons\nGodot_v4.4"
printf "\n1) Attempting to install opencv shared objects"

docker create --pull missing --name ar-docker-export kauzoo/ar-exporter:latest
echo Exporting to $EXPORT_DIR
echo Exporting .so to $EXPORT_DIR/lib
mkdir opencv
docker cp ar-docker-export:/home/ardocker/export/opencv/lib/ $EXPORT_DIR/lib
echo Exporting headers to $EXPORT_DIR/include
docker cp ar-docker-export:/home/ardocker/export/opencv/include/opencv4/ $EXPORT_DIR/include
docker rm ar-docker-export

printf "2) Attempting to add opencv shared objects to project"
cp -r $EXPORT_DIR/lib/*.so* demo/bin/

printf "3) Attempting to build godot-cpp"
cd godot-cpp
git submodule update --init
scons platform=$PLATFORM custom_api_file=$EXTENSION_API_PATH
cd ..

printf "4) Generating compiledb.json"
scons compiledb=yes

printf "5) Attempting to build gdextension"
scons platform=linux
printf "Done"
