#!/bin/bash
OPENCV_VERSION=4.9.0
OPENCV_BUILD_LIST=core,imgcodecs,imgproc,videoio,objdetect,video,tracking
PLATFORM=linux
EXTENSION_API_PATH=../gdextension-api-files/gd4.4_linux_extension_api.json


echo "PREREQUISITES"
echo "- CPP-COMPILER: g++"
echo "- BUILD TOOLS: scons, cmake"
echo "- OTHER: wget, git"
echo "- LIBRARIES: ffmpeg, libavcodec"


echo "STEP 1 : BUILD OPENCV"
wget -v -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/$OPENCV_VERSION.zip
wget -v -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/$OPENCV_VERSION.zip
unzip opencv.zip && unzip opencv_contrib.zip && rm -vf opencv.zip && rm -vf opencv_contrib.zip
mkdir {build,opencv}
cd build
cmake -DCMAKE_INSTALL_PREFIX=../opencv -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-$OPENCV_VERSION/modules -DBUILD_LIST=$OPENCV_BUILD_LIST -DWITH_FFMPEG=ON ../opencv-$OPENCV_VERSION
if [ $(cmake -LAH | grep -c ".*FFMPEG:.*YES") -nw 1 ]; then
	echo "WARNING: FAILED TO PROPERLY BUILD WITH FFMPEG"
	exit
fi
cmake --build . --target install
cd ..
rm -vrf opencv_contrib-$OPENCV_VERSION && rm -vrf opencv-$OPENCV_VERSION

echo "STEP 2 : ADD libopencv_*.so* TO demo/bin"
cp -v -r opencv/lib/libopencv_* demo/bin/

echo "STEP 3 : BUILD godot-cpp"
cd godot-cpp
git submodule update --init
scons platform=$PLATFORM custom_api_file=$EXTENSION_API_PATH
cd ..

echo "STEP 4 : GENERATE compiledb.json"
scons compiledb=yes

echo "STEP 5 : BUILD gdextension"
scons platform=$PLATFORM
echo Done
