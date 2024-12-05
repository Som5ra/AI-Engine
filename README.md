<!-- ## Android Build -->

<!-- ### Prepare
1. Android NDK
2. OpenCV 4.10 Android SDK (refer to opencv_android_sdk_tree_reference.txt)
   - copy /OpenCV-android-sdk/sdk/native/libs/${ANDROID_ABI}/libopencv_java4.so to the folder where your native engine is at. -->


### Build

```
Linux Host:
python3 build.py --android --linux --windows (--noinstall)

MACOS Host:
python3 build.py --macos --ios (--noinstall)
```


### NOTES
Refer to Notion for [docs](https://www.notion.so/gustolabs/AI-Engine-Build-Process-13b5f7c72a4a80b0b8c4e3a31933caa3)


### Some model export guide:
Refer to [docs](https://github.com/Som5ra/AI-Engine/blob/main/model_tools/export_onnx_mmdetection.md)