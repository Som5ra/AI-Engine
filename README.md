## Intro 👇👇
👏👋This is a lite library with pure C++ for AI model inference, aiming to deploy on mobile devices easily. This can be built to be migrated into **Unity** and used by C#.🎉🎉

## NN Inference (Developing) 👇👇

Strongly depending on module: onnxruntime.

**Click to Check Demo**

1. **2D detection Model**
    1. RTMDet-series
    2. YOLO-series
2. [Face Landmark Model](https://github.com/Som5ra/AI-Engine/blob/main/media/demo/face_geometry_demo.gif)
    1. Face Detector
    2. Face Landmarker
3. [Human Segmentation Model](https://github.com/Som5ra/AI-Engine/blob/main/media/demo/human_segmentation_demo.gif)
    1. Selfie (Close to camera)
4. **Human Pose Model**
    <!-- 1. RTMO `Far Scenario`   `Single Stage` -->
    <!-- 2. VIT Pose `Pending` -->
    1. RTMPOSE
### Supported ONNXRuntime Execution Providers:
|         | Linux | Android (exclude x86) | MacOS     | IOS       | Windows | WebAssembly |
|---------|-------|-----------------------|-----------|-----------|---------|---------|
| CPU     | ✅     | ✅                     | ✅         | ✅         | ✅       | ✅       |
| GPU     | -     | -                     | ✅(CoreML) | ✅(CoreML) | -       | -       |
| XNNPACK | ✅     | ✅                     | -         | -         | ✅       | -       |
| NNAPI   | -     | ✅                     | -         | -         | -       | -       |

## Other Supported Modules:

### Post-processing

1. [Non-maximum Suppression](https://www.notion.so/Post-Processing-NMS-13b5f7c72a4a804b8751ea6bf1272c3c?pvs=21)
2. [Face-Geometry](https://www.notion.so/Post-Processing-Face-Geometry-13b5f7c72a4a809bbd5cdc7ccfea48ca?pvs=21)

|                         | Linux | Android | MacOS | IOS | Windows |
|-------------------------|-------|---------|-------|-----|---------|
| Non-maximum Suppression | ✅     | ✅       | ✅     | ✅   | ✅       |
| Face-Geometry           | ✅     | ✅       | ✅     | ✅   | ✅       |


### Supported 3rd parties:
|                             | Linux |     Android    | MacOS | IOS | Windows | WebAssembly |
|:---------------------------:|:-----:|:--------------:|:-----:|:---:|:-------:|:-------:|
|       OpenCV - Mobile       | ✅     | ✅              | ✅     | ✅   | ✅       | ✅       |
|         ONNXRuntime         | ✅     | ✅ excluding x86 | ✅     | ✅   | ✅       | ✅       |
| nlohmann json (header only) | ✅     | ✅              | ✅     | ✅   | ✅       | ✅       |
|     Eigen (header only)     | ✅     | ✅              | ✅     | ✅   | ✅       | ✅       |
|            OpenMP           | ✅     | ✅              | ✅     | ✅   | ✅       | ✅       |

### Build

```
Linux Host:
./build_linux.sh install
# python3 build.py --android --linux

MACOS Host:
./build_osx.sh
./build_ios.sh
# python3 build.py --macos --ios (--noinstall)

Windows Host (with vs2022):
# python3 build.py --windows (--noinstall)
```

### NOTES
Refer to Notion for [detail documentation](https://www.notion.so/gustolabs/AI-Engine-Build-Process-13b5f7c72a4a80b0b8c4e3a31933caa3)

### Some model export guide:
Refer to [docs](https://github.com/Som5ra/AI-Engine/blob/main/model_tools/export_onnx_mmdetection.md)