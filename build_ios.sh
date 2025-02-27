#!/usr/bin/env bash

set -e

TARGET_OS="ios"
ONNXRUNTIME_SOURCE_URL="https://github.com/Som5ra/AI-Engine/releases/download/3rd-party/onnxruntime.xcframework-1.19.2.zip"
OPENCV_SOURCE_URL="https://github.com/Som5ra/AI-Engine/releases/download/3rd-party/opencv-mobile-4.10.0-ios.zip"

# check if 3rdparty/onnxruntime/onnxruntime.xcframework-1.19.2 exists
if [ ! -d "3rdparty/onnxruntime/onnxruntime.xcframework-1.19.2" ]; then
    echo "No Prebuilt onnxruntime found, Downloading from $ONNXRUNTIME_SOURCE_URL"
    cd 3rdparty/onnxruntime
    wget $ONNXRUNTIME_SOURCE_URL -O onnxruntime.xcframework-1.19.2.zip
    unzip onnxruntime.xcframework-1.19.2.zip
    rm onnxruntime.xcframework-1.19.2.zip
    cd ../../
else
    echo "Prebuilt onnxruntime found: 3rdparty/onnxruntime/onnxruntime.xcframework-1.19.2"
fi

# check if 3rdparty/opencv/opencv-mobile-4.10.0-ios exists
if [ ! -d "3rdparty/opencv/opencv-mobile-4.10.0-ios" ]; then
    echo "No Prebuilt opencv found, Downloading from $OPENCV_SOURCE_URL"
    cd 3rdparty/opencv
    wget $OPENCV_SOURCE_URL -O opencv-mobile-4.10.0-ios.zip
    unzip opencv-mobile-4.10.0-ios.zip
    rm opencv-mobile-4.10.0-ios.zip
    cd ../../
else
    echo "Prebuilt OpenCV found: 3rdparty/opencv/opencv-mobile-4.10.0-ios"
fi



cmake -DBUILD_PLATFORM=$TARGET_OS \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -GXcode \
      -DCMAKE_TOOLCHAIN_FILE=ios.toolchain.cmake \
      -DPLATFORM=OS64 \
      -DENABLE_ARC=1 \
      -DENABLE_VISIBILITY=0 \
      -DCMAKE_INSTALL_PREFIX=./install \
    -DBUILD_SHARED_LIBS=OFF \
      -S . -B build/$TARGET_OS 

cmake --build build/$TARGET_OS -j8 --config Release


if [ "$1" == "install" ]; then
    cmake --install build/$TARGET_OS
fi

