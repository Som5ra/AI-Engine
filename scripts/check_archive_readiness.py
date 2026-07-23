#!/usr/bin/env python3
"""Run dependency-free structural checks for the archival codebase."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRECTORIES = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "3rdparty",
    "build",
    "media",
    "third_party",
    "weights",
}


def maintained_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.relative_to(ROOT).parts):
            continue
        files.append(path)
    return files


def read_text(relative_path: str) -> str:
    path = ROOT / relative_path
    if not path.is_file():
        raise AssertionError(f"required file is missing: {relative_path}")
    return path.read_text(encoding="utf-8")


def require_text(relative_path: str, expected: str) -> None:
    if expected not in read_text(relative_path):
        raise AssertionError(
            f"{relative_path} does not contain required invariant: {expected}"
        )


def reject_bytes(files: list[Path], forbidden: bytes, label: str) -> None:
    hits = []
    for path in files:
        if forbidden.lower() in path.read_bytes().lower():
            hits.append(str(path.relative_to(ROOT)))
    if hits:
        raise AssertionError(f"{label} found in: {', '.join(sorted(hits))}")


def main() -> int:
    files = maintained_files()

    retired_identifier = bytes.fromhex("677573746f")
    developer_media_path = bytes.fromhex("2f6d656469612f736f6d6272616c69")
    developer_home_path = bytes.fromhex("2f686f6d652f736f6d6272616c69")
    debugger_trap = bytes.fromhex("5f5f6465627567627265616b")
    missing_openmp_shim = bytes.fromhex("73696d706c656f6d70")
    reject_bytes(files, retired_identifier, "retired identifier")
    reject_bytes(files, developer_media_path, "developer media path")
    reject_bytes(files, developer_home_path, "developer home path")
    reject_bytes(files, debugger_trap, "interactive debugger trap")
    reject_bytes(files, missing_openmp_shim, "missing simple OpenMP shim")

    require_text(
        "include/utils.h",
        "constexpr int ERR_OK = 0x00000000;",
    )
    require_text("include/utils.h", "#define CUSTOM_API")
    require_text(
        "src/two_stage_human_pose_extractor_2d.cc",
        "detect_interval_(std::max(1, detect_interval))",
    )
    require_text(
        "src/two_stage_human_pose_extractor_2d.cc",
        "pose_results_.resize(detection_result_->boxes.size());",
    )
    require_text(
        "src/multi_stage_face_geometry_3d.cc",
        "multi_face_landmarks.reserve(detector_result->boxes.size());",
    )
    require_text(
        "tools/face_geometry/calculator.cc",
        "static_cast<std::size_t>(face_index) *",
    )
    require_text(
        "tools/face_geometry/calculator.cc",
        "face_geometries.size() !=",
    )
    require_text("src_unity_api/unity_api.h", "Custom_Model_Destroy")
    require_text(
        "src_unity_api/unity_api.h",
        "Custom_Human_Pose_Pipeline_Destroy",
    )
    require_text(
        "src_wasm_api/wasm_api.cc",
        "Custom_Face_Geometry_Pipeline_Destroy",
    )
    require_text(
        "include/tools/face_geometry/calculator.h",
        "face_mesh_calculator_destroy",
    )
    require_text(
        "CMakeLists.txt",
        'option(AI_ENGINE_BUILD_6D_TRACKING "Build the experimental 6D tracking tools" OFF)',
    )
    require_text(
        "CMakeLists.txt",
        '"${CMAKE_SOURCE_DIR}/src/multi_stage_face_geometry_3d.cc"',
    )
    require_text(
        "tools/face_geometry/geometry_pipeline.cc",
        "std::make_pair(std::move(multi_face_geometry), ret_signal)",
    )
    require_text(
        "tools/face_geometry/geometry_pipeline.cc",
        "Layout::ROW_MAJOR",
    )
    require_text(
        "tools/face_geometry/face_geometry.cc",
        "canonical_mesh.canonical_mesh_num_vertices != 478",
    )
    require_text(
        "tools/face_geometry/face_geometry.cc",
        'data.at("input_source").template get<InputSource>()',
    )
    require_text(
        "build.py",
        'ANDROID_ABIS = ("armeabi-v7a", "arm64-v8a", "x86_64")',
    )
    require_text(
        "cmake/Dependencies.cmake",
        "set(AI_ENGINE_ANDROID_ABIS armeabi-v7a arm64-v8a x86_64)",
    )
    require_text(
        "src/multi_stage_face_geometry_3d.cc",
        "detect_interval_(std::max(1, detect_interval))",
    )
    require_text(
        "src/multi_stage_face_geometry_3d.cc",
        "DrawCoordinateAxes(",
    )
    require_text(
        "src/BaseONNX.cc",
        "Model preprocessing configuration does not match the input tensor",
    )
    require_text(
        "src/detector2d_family.cc",
        "index * detection_stride",
    )
    require_text(
        "src/human_pose_family.cc",
        "RTMPose output shapes must be [1, joints, bins]",
    )

    for example_cmake in (
        "examples/face_geometry_example/CMakeLists.txt",
        "examples/detection_2d_example/CMakeLists.txt",
        "examples/ort/CMakeLists.txt",
        "examples/json_test/CMakeLists.txt",
    ):
        if "OpenMP::OpenMP_CXX" in read_text(example_cmake):
            raise AssertionError(
                f"{example_cmake} bypasses the optional OpenMP target"
            )

    removed_paths = [
        "tools/6d_tracking/build",
        "include/detector2d.h_deprecated",
        "include/json.hpp_deprecated",
        "src/detector2d.cpp_deprecated",
    ]
    unexpected = [path for path in removed_paths if (ROOT / path).exists()]
    if unexpected:
        raise AssertionError(
            "generated or deprecated paths still exist: " + ", ".join(unexpected)
        )

    print(
        f"Archive readiness checks passed: {len(files)} maintained files; "
        "all reviewed structural invariants hold."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"Archive readiness check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
