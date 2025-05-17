#!/bin/sh
EXTENSION_API_PATH=../gdextension-api-files/gd4.4_linux_extension_api.json
PLATFORM=linux
EXPORT_DIR=opencv
EXPORTIMAGE=kauzoo/opencvbuild:latest
echo Running default linux install
echo REQUIREMENTS
printf "Docker (installed and running)\nScons\nGodot_v4.4\n"
printf "\n1) Attempting to install opencv shared objects\n"

docker create --pull missing --name opencvexport $EXPORTIMAGE
echo "Exporting to $EXPORT_DIR"
docker cp opencvexport:/opencv/. $EXPORT_DIR
docker rm opencvexport

echo "2) Attempting to add opencv libraries to project"
cp -r $EXPORT_DIR/lib/*.a demo/bin/
cp -r $EXPORT_DIR/lib/opencv4/3rdparty/*.a demo/bin

echo "3) Attempting to build godot-cpp"
cd godot-cpp
git submodule update --init
scons platform=$PLATFORM custom_api_file=$EXTENSION_API_PATH
cd ..

echo "4) Generating compiledb.json"
scons compiledb=yes

echo "5) Attempting to build gdextension"
scons platform=linux
echo Done
