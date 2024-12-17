#!/usr/bin/env bash

set -e

TARGET_OS="macos"
ONNXRUNTIME_SOURCE_URL="https://github.com/Som5ra/AI-Engine/releases/download/3rd-party/onnxruntime-osx-arm64-static_lib-1.19.2.zip"
OPENCV_SOURCE_URL="https://github.com/Som5ra/AI-Engine/releases/download/3rd-party/opencv-mobile-4.10.0-macos.zip"

# check if 3rdparty/onnxruntime/onnxruntime-osx-arm64-static_lib-1.19.2 exists
if [ ! -d "3rdparty/onnxruntime/onnxruntime-osx-arm64-static_lib-1.19.2" ]; then
    echo "No Prebuilt onnxruntime found, Downloading from $ONNXRUNTIME_SOURCE_URL"
    cd 3rdparty/onnxruntime
    wget $ONNXRUNTIME_SOURCE_URL -O onnxruntime-osx-arm64-static_lib-1.19.2.zip
    unzip onnxruntime-osx-x64-static_lib-1.19.2.zip
    rm onnxruntime-osx-arm64-static_lib-1.19.2.zip
    cd ../../
else
    echo "Prebuilt onnxruntime found: 3rdparty/onnxruntime/onnxruntime-osx-arm64-static_lib-1.19.2"
fi

# check if 3rdparty/opencv/opencv-mobile-4.10.0-macos exists
if [ ! -d "3rdparty/opencv/opencv-mobile-4.10.0-macos" ]; then
    echo "No Prebuilt opencv found, Downloading from $OPENCV_SOURCE_URL"
    cd 3rdparty/opencv
    wget $OPENCV_SOURCE_URL -O opencv-mobile-4.10.0-macos.zip
    unzip opencv-mobile-4.10.0-ubuntu-2204.zip
    rm opencv-mobile-4.10.0-macos.zip
    cd ../../
else
    echo "Prebuilt OpenCV found: 3rdparty/opencv/opencv-mobile-4.10.0-macos"
fi


cmake -DBUILD_PLATFORM=$TARGET_OS \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_TOOLCHAIN_FILE=osx.toolchain.cmake \
      -S . -B build/$TARGET_OS 

cmake --build build/$TARGET_OS -j8


if [ "$1" == "install" ]; then
    cmake --install build/$TARGET_OS
fi

