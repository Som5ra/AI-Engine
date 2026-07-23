# AI-Engine 📦

AI-Engine is a C++17 inference library for ONNX models on Linux, Android,
macOS, iOS, Windows, and WebAssembly. It includes native and Unity-facing
bindings for:

- 2D object detection
- face detection, landmarks, and 3D face geometry
- human segmentation
- multi-person 2D pose estimation

This repository is being prepared as a stable archive. See
[ARCHIVE.md](ARCHIVE.md) for the handoff status, known limitations, and final
archive checklist.

## Supported model families

| Capability | Supported models / pipeline | Status |
|---|---|---|
| 2D object detection | RTMDet series, YOLO series | Maintained |
| Face detection | MediaPipe face detector | Maintained |
| Face landmarks | MediaPipe face landmarker, including iris landmarks | Maintained |
| 3D face geometry | Multi-face metric geometry and pose matrices | Maintained |
| Human segmentation | Selfie segmentation for close-camera scenarios | Maintained |
| Human pose | RTMPose with multi-person detection | Maintained |
| 6D object tracking | Region-based CAD model tracker | Experimental; opt-in |

Demo assets are retained for
[face geometry](media/demo/face_geometry_demo.gif) and
[human segmentation](media/demo/human_segmentation_demo.gif).

## Platform and architecture support

### ONNX Runtime execution providers

| Execution provider | Linux | Android | macOS | iOS | Windows | WebAssembly |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| CPU | Yes | Yes | Yes | Yes | Yes | Yes |
| CoreML | — | — | Yes | Yes | — | — |
| XNNPACK | Yes | Yes | — | — | Yes | — |
| NNAPI | — | Yes | — | — | — | — |

Android support excludes the legacy x86 ONNX Runtime target; `armeabi-v7a`,
`arm64-v8a`, and `x86_64` remain available in the build driver.

### Post-processing modules

| Module | Linux | Android | macOS | iOS | Windows | WebAssembly |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Non-maximum suppression | Yes | Yes | Yes | Yes | Yes | Yes |
| Multi-face geometry | Yes | Yes | Yes | Yes | Yes | Yes |

### Third-party architecture matrix

| Dependency | Linux | Android | macOS | iOS | Windows | WebAssembly |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| OpenCV Mobile 4.10.0 | Yes | Yes | Yes | Yes | Yes | Yes |
| ONNX Runtime | Yes | Yes, except x86 | Yes | Yes | Yes | Yes |
| nlohmann/json 3.11 | Yes | Yes | Yes | Yes | Yes | Yes |
| Eigen 3.4 | Yes | Yes | Yes | Yes | Yes | Yes |
| OpenMP acceleration | Yes | Toolchain-dependent | Yes | — | — | — |

These tables describe the retained target configuration. The archive checklist
still requires native smoke builds for every platform that must remain
supportable.

## Public API names

The maintained public surface uses neutral `Custom` naming:

- native library: `CustomEngine`
- Unity library: `CustomEngineUnity`
- WebAssembly target: `CustomEngineWASM`
- C exports: `Custom_*`
- status and geometry types: `CustomStatus` and `CustomRect`
- internal namespaces: `custom_*`

This is an intentional breaking rename. Consumers must update their native
library names and exported-function declarations together.

## Dependencies

The default build requires:

- CMake 3.20 or newer
- a C++17 compiler
- OpenCV Mobile 4.10.0
- ONNX Runtime prebuilt archives referenced by the platform bootstrap scripts
- vendored Eigen 3.4 and nlohmann/json 3.11
- OpenMP where the target toolchain provides it

The default configuration builds only the native engine and Unity bindings.
Examples, standalone tools, and experimental 6D tracking are opt-in.

## Build

Linux dependencies can be bootstrapped and compiled with:

```bash
./build_linux.sh
./build_linux.sh install
```

The cross-platform Python driver assumes the platform dependencies are already
present:

```bash
python3 build.py linux --install
python3 build.py macos --toolchain osx.toolchain.cmake
python3 build.py ios --toolchain ios.toolchain.cmake --generator Xcode
python3 build.py windows --toolchain win.toolchain.cmake
python3 build.py android \
  --toolchain /path/to/android-ndk/build/cmake/android.toolchain.cmake \
  --android-abi arm64-v8a
```

Linux and macOS also have CMake presets:

```bash
cmake --preset linux-release
cmake --build --preset linux-release
```

Optional components are enabled explicitly:

```bash
python3 build.py linux \
  --cmake-arg=-DAI_ENGINE_BUILD_EXAMPLES=ON \
  --cmake-arg=-DAI_ENGINE_BUILD_STANDALONE_TOOLS=ON
```

Experimental 6D tracking additionally requires OpenGL, GLEW, and GLFW:

```bash
python3 build.py linux --cmake-arg=-DAI_ENGINE_BUILD_6D_TRACKING=ON
```

## Archive readiness check

Run the dependency-free structural regression check before merging or
archiving:

```bash
python3 scripts/check_archive_readiness.py
```

The check covers the P0 logic fixes, ownership APIs, portable build defaults,
identifier cleanup, and removal of generated/deprecated artifacts.

## Model export notes

The historical MMDetection/MMYOLO export notes are retained in
[model_tools/export_onnx_mmdetection.md](model_tools/export_onnx_mmdetection.md).
