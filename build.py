#!/usr/bin/env python3
"""Configure, build, and optionally install AI-Engine."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


SUPPORTED_PLATFORMS = ("linux", "android", "macos", "ios", "windows", "wasm")
ANDROID_ABIS = ("armeabi-v7a", "arm64-v8a", "x86", "x86_64")


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("platform", choices=SUPPORTED_PLATFORMS)
    parser.add_argument("--build-type", default="Release")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--generator")
    parser.add_argument("--android-abi", choices=ANDROID_ABIS, default="arm64-v8a")
    parser.add_argument("--android-platform", default="android-26")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--install-prefix", type=Path)
    parser.add_argument(
        "--cmake-arg",
        action="append",
        default=[],
        help="Additional configure argument; may be supplied more than once",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be greater than zero")

    suffix = args.android_abi if args.platform == "android" else None
    default_build_dir = Path("build") / args.platform
    if suffix:
        default_build_dir /= suffix
    build_dir = args.build_dir or default_build_dir

    configure = [
        "cmake",
        "-S",
        ".",
        "-B",
        str(build_dir),
        f"-DBUILD_PLATFORM={args.platform}",
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
    ]

    if args.generator:
        configure.extend(["-G", args.generator])
    if args.toolchain:
        configure.append(f"-DCMAKE_TOOLCHAIN_FILE={args.toolchain}")
    if args.install_prefix:
        configure.append(f"-DCMAKE_INSTALL_PREFIX={args.install_prefix}")

    if args.platform == "android":
        if not args.toolchain:
            raise SystemExit("Android builds require --toolchain <android.toolchain.cmake>")
        configure.extend(
            [
                f"-DANDROID_ABI={args.android_abi}",
                f"-DANDROID_PLATFORM={args.android_platform}",
            ]
        )
    elif args.platform == "ios":
        if not args.toolchain:
            raise SystemExit("iOS builds require --toolchain <ios.toolchain.cmake>")
        configure.extend(["-DPLATFORM=OS64", "-DENABLE_ARC=1"])
    elif args.platform == "wasm" and not args.toolchain:
        raise SystemExit(
            "WASM builds require --toolchain <Emscripten.cmake> "
            "or invocation through emcmake"
        )

    configure.extend(args.cmake_arg)
    run(configure)

    build = [
        "cmake",
        "--build",
        str(build_dir),
        "--config",
        args.build_type,
        "--parallel",
        str(args.jobs),
    ]
    run(build)

    if args.install:
        run(
            [
                "cmake",
                "--install",
                str(build_dir),
                "--config",
                args.build_type,
            ]
        )


if __name__ == "__main__":
    main()
