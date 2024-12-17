#!/usr/bin/env bash

set -e

TARGET_OS="linux"
ONNXRUNTIME_SOURCE_URL="https://github.com/Som5ra/AI-Engine/releases/download/3rd-party/onnxruntime-linux-x64-static_lib-1.19.2.zip"
OPENCV_SOURCE_URL="https://github.com/Som5ra/AI-Engine/releases/download/3rd-party/opencv-mobile-4.10.0-ubuntu-2204.zip"

# check if 3rdparty/onnxruntime/onnxruntime-linux-x64-static_lib-1.19.2 exists
if [ ! -d "3rdparty/onnxruntime/onnxruntime-linux-x64-static_lib-1.19.2" ]; then
    echo "No Prebuilt onnxruntime found, Downloading from $ONNXRUNTIME_SOURCE_URL"
    cd 3rdparty/onnxruntime
    wget $ONNXRUNTIME_SOURCE_URL -O onnxruntime-linux-x64-static_lib-1.19.2.zip
    unzip onnxruntime-linux-x64-static_lib-1.19.2.zip
    rm onnxruntime-linux-x64-static_lib-1.19.2.zip
    cd ../../
else
    echo "Prebuilt onnxruntime found: 3rdparty/onnxruntime/onnxruntime-linux-x64-static_lib-1.19.2"
fi

# check if 3rdparty/opencv/opencv-mobile-4.10.0-ubuntu-2204 exists
if [ ! -d "3rdparty/opencv/opencv-mobile-4.10.0-ubuntu-2204" ]; then
    echo "No Prebuilt opencv found, Downloading from $OPENCV_SOURCE_URL"
    cd 3rdparty/opencv
    wget $OPENCV_SOURCE_URL -O opencv-mobile-4.10.0-ubuntu-2204.zip
    unzip opencv-mobile-4.10.0-ubuntu-2204.zip
    rm opencv-mobile-4.10.0-ubuntu-2204.zip
    cd ../../
else
    echo "Prebuilt OpenCV found: 3rdparty/opencv/opencv-mobile-4.10.0-ubuntu-2204"
fi


cmake -DBUILD_PLATFORM=$TARGET_OS \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -S . -B build/$TARGET_OS 

cmake --build build/$TARGET_OS -j8


if [ "$1" == "install" ]; then
    cmake --install build/$TARGET_OS
fi

