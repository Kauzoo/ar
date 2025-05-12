#!/bin/sh
GDPROJECT=demo
EXPORT_DIR=./$GDPROJECT/bin
docker create --pull missing --name ar-docker-export kauzoo/ar-docker:latest
echo Exporting to $EXPORT_DIR
docker cp ar-docker-export:/home/ardocker/export/opencv/lib/ $EXPORT_DIR
cp -r $EXPORT_DIR/lib/libopencv_* $EXPORT_DIR/
rm -rf $EXPORT_DIR/lib
docker rm ar-docker-export