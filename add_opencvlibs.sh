#!/bin/sh
GDPROJECT=demo
EXPORT_DIR2=./$GDPROJECT/bin
docker create --pull missing --name ar-docker-export kauzoo/ar-exporter:latest
echo Exporting to $EXPORT_DIR2
docker cp ar-docker-export:/home/ardocker/export/opencv/lib/ $EXPORT_DIR2
docker cp ar-docker-export:/home/ardocker/export/libavcodec.so.60 $EXPORT_DIR2
docker cp ar-docker-export:/home/ardocker/export/libavcodec.so.60.31.102 $EXPORT_DIR2
cp -r $EXPORT_DIR2/lib/libopencv_* $EXPORT_DIR2/
rm -rf $EXPORT_DIR2/lib
docker rm ar-docker-export