## Android Build

<!-- ### Prepare
1. Android NDK
2. OpenCV 4.10 Android SDK (refer to opencv_android_sdk_tree_reference.txt)
   - copy /OpenCV-android-sdk/sdk/native/libs/${ANDROID_ABI}/libopencv_java4.so to the folder where your native engine is at. -->


### Build

```
python3 build.py --android --linux
python3 build.py --macos
python3 build.py --windows
```