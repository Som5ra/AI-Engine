## Android Build

### Prepare
1. Android NDK
2. OpenCV 4.10 Android SDK (refer to opencv_android_sdk_tree_reference.txt)
   - copy /OpenCV-android-sdk/sdk/native/libs/${ANDROID_ABI}/libopencv_java4.so to the folder where your native engine is at.


### Build

```
mkdir build
cd build

cmake -DCMAKE_TOOLCHAIN_FILE=/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-22 ../
make -j8
make install
```