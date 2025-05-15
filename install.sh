#!/bin/sh
printf "Running default linux install"
printf "REQUIREMENTS"
printf "Docker (installed and running)\nScons\nGodot_v4.4"
printf "\n1) Attempting to install opencv shared objects"
sh ./install_opencv.sh
printf "2) Attempting to add opencv shared objects to project"
sh ./add_opencvlibs.sh
printf "3) Attempting to build godot-cpp"
cd ./godot-cpp
git submodule update --init
cd ..
sh ./build_gdcppbindings.sh
printf "4) Generating compiledb.json"
scons compiledb=yes
printf "5) Attempting to build gdextension"
scons platform=linux
printf "Done"
