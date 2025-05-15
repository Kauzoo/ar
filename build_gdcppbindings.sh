#!/bin/sh
EXTENSION_API_PATH=../gdextension-api-files/gd4.4_linux_extension_api.json
PLATFORM=linux
cd godot-cpp
scons platform=$PLATFORM custom_api_file=$EXTENSION_API_PATH
