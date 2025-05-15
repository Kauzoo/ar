#!/bin/sh
EXPORT_DIR=./opencv
docker create --pull missing --name ar-docker-export kauzoo/ar-exporter:latest
echo Exporting to $EXPORT_DIR
echo Exporting .so to $EXPORT_DIR/lib
docker cp ar-docker-export:/home/ardocker/export/opencv/lib/ $EXPORT_DIR
echo Exporting headers to $EXPORT_DIR/include
docker cp ar-docker-export:/home/ardocker/export/opencv/include/ $EXPORT_DIR
docker rm ar-docker-export