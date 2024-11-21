import os
import subprocess
import argparse
import shutil
import logging

logging.basicConfig(level=logging.INFO)


def execute(cmd, shell=False):
    try:
        logging.debug("Executing: %s" % cmd)
        logging.info('Executing: ' + ' '.join(cmd))
        retcode = subprocess.call(cmd, shell=shell)
        if retcode < 0:
            raise Exception("Child was terminated by signal: %s" % -retcode)
        elif retcode > 0:
            raise Exception("Child returned: %s" % retcode)
    except OSError as e:
        raise Exception("Execution failed: %d / %s" % (e.errno, e.strerror))
    
def build_macos(toolchain = 'osx.toolchain.cmake', install = True):
    binary_dir = 'build/build-macos'

    compile_cmd = ['cmake']
    compile_cmd.append('-DBUILD_PLATFORM=macos')
    compile_cmd.append(f'-DCMAKE_TOOLCHAIN_FILE={toolchain}')
    compile_cmd.append('-DCMAKE_BUILD_TYPE=Release')
    compile_cmd.append('-DCMAKE_EXPORT_COMPILE_COMMANDS=ON')

    compile_cmd.append('-S .')
    compile_cmd.append(f'-B {binary_dir}')
    execute(compile_cmd)
    
    build_cmd = [f'cmake --build {binary_dir}']
    build_cmd.append('-j8')
    execute(build_cmd, shell=True)

    if install:
        install_cmd = [f'cmake --install {binary_dir}']
        execute(install_cmd, shell=True)

def build_linux(install = True):
    binary_dir = 'build/build-linux'

    compile_cmd = ['cmake']
    compile_cmd.append('-DBUILD_PLATFORM=linux')
    compile_cmd.append('-DCMAKE_BUILD_TYPE=Release')
    compile_cmd.append('-DCMAKE_EXPORT_COMPILE_COMMANDS=ON')
    # compile_cmd.append('-DCMAKE_CXX_STANDARD=14')

    compile_cmd.append('-S .')
    compile_cmd.append(f'-B {binary_dir}')
    execute(compile_cmd)
    
    build_cmd = [f'cmake --build {binary_dir}']
    build_cmd.append('-j8')
    execute(build_cmd, shell=True)

    if install:
        install_cmd = [f'cmake --install {binary_dir}']
        execute(install_cmd, shell=True)


def build_windows(toolchain = 'win.toolchain.cmake', install = True):
    binary_dir = 'build/build-windows'
    # Ensure the build directory exists
    if not os.path.exists(binary_dir):
        os.makedirs(binary_dir)

    compile_cmd = ['cmake']
    compile_cmd.append('-DBUILD_PLATFORM=windows')
    compile_cmd.append(f'-DCMAKE_TOOLCHAIN_FILE={toolchain}')
    compile_cmd.append('-DCMAKE_BUILD_TYPE=Release')
    compile_cmd.append('-DCMAKE_EXPORT_COMPILE_COMMANDS=ON')

    # compile_cmd.append('-DCMAKE_TOOLCHAIN_FILE=~/mingw-w64-x86_64.cmake')
    # compile_cmd.append('-DCMAKE_EXPORT_COMPILE_COMMANDS=ON')
    # compile_cmd.append('-DCMAKE_MAKE_PROGRAM=mingw32-make')
    # compile_cmd.append('-G MinGW Makefiles')
    
    compile_cmd.append('-S .')
    compile_cmd.append(f'-B {binary_dir}')
    execute(compile_cmd)
    
    # windows_make_path = binary_dir.replace("/", "\\")
    build_cmd = ['cmake', '--build', binary_dir, '-j8']
    execute(build_cmd)

    if install:
        install_cmd = [f'cmake --install {binary_dir}']
        execute(install_cmd, shell=True)

def build_android(
        toolchain = "/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/build/cmake/android.toolchain.cmake",
        ANDROID_ABI = 'arm64-v8a', 
        ANDROID_PLATFORM = 'android-22',
        install = True):
    
    android_abi_enum = ['armeabi-v7a', 'arm64-v8a', 'x86', 'x86_64']
    binary_dir = f'build/build-android/{ANDROID_ABI}'

    if ANDROID_ABI not in android_abi_enum:
        raise Exception('Invalid ANDROID_ABI')

    compile_cmd = ['cmake']
    compile_cmd.append('-DBUILD_PLATFORM=android')
    compile_cmd.append('-DCMAKE_BUILD_TYPE=Release')        
    compile_cmd.append('-DCMAKE_EXPORT_COMPILE_COMMANDS=ON')


    compile_cmd.append(f'-DCMAKE_TOOLCHAIN_FILE={toolchain}')
    compile_cmd.append(f'-DANDROID_ABI={ANDROID_ABI}')
    compile_cmd.append(f'-DANDROID_PLATFORM={ANDROID_PLATFORM}')


    compile_cmd.append('-S .')
    compile_cmd.append(f'-B {binary_dir}')
    execute(compile_cmd)
    
    build_cmd = [f'cmake --build {binary_dir}']
    build_cmd.append('-j8')
    execute(build_cmd, shell=True)

    if install:
        install_cmd = [f'cmake --install {binary_dir}']
        execute(install_cmd, shell=True)

def build_ios(toolchain = 'ios.toolchain.cmake', install = True):
    # cmake -B build -G Xcode -DCMAKE_TOOLCHAIN_FILE=../../../ios-cmake/ios.toolchain.cmake -DPLATFORM=OS64 -DBUILD_PLATFORM=ios -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    # cmake --build build --config Release

    binary_dir = f'build/build-ios/'

    compile_cmd = ['cmake']
    compile_cmd.append('-DBUILD_PLATFORM=ios')
    compile_cmd.append('-DCMAKE_BUILD_TYPE=Release')        
    compile_cmd.append('-DCMAKE_EXPORT_COMPILE_COMMANDS=ON')

    compile_cmd.append(f'-G Xcode')
    compile_cmd.append(f'-DCMAKE_TOOLCHAIN_FILE={toolchain}')
    compile_cmd.append(f'-DPLATFORM=OS64')
    compile_cmd.append(f'-DENABLE_ARC=1')
    compile_cmd.append(f'-DENABLE_VISIBILITY=0')
    compile_cmd.append(f'-DCMAKE_INSTALL_PREFIX=./install')
    compile_cmd.append(f'-DBUILD_SHARED_LIBS=OFF')
    


    compile_cmd.append('-S .')
    compile_cmd.append(f'-B {binary_dir}')
    execute(compile_cmd)
    
    build_cmd = [f'cmake --build {binary_dir} --config Release']
    build_cmd.append('-j8')
    execute(build_cmd, shell=True)

    if install:
        install_cmd = [f'cmake --install {binary_dir}']
        execute(install_cmd, shell=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build the project')
    parser.add_argument('--android', action='store_true')
    parser.add_argument('--macos', action='store_true')
    parser.add_argument('--linux', action='store_true')
    parser.add_argument('--windows', action='store_true')
    parser.add_argument('--ios', action='store_true')
    parser.add_argument('--noinstall', action='store_true')

    args = parser.parse_args()

    _install = False if args.noinstall else True

    if args.linux: 
        build_linux(install = _install)

    if args.android:
        abis = ['armeabi-v7a', 'arm64-v8a', 'x86', 'x86_64']
        for abi in abis:
            build_android(ANDROID_ABI=abi, install = _install)
        # build_android(install = _install)
    
    if args.macos:
        build_macos(install = _install)


    if args.windows:
        build_windows(install = _install)

    if args.ios:
        build_ios(install = _install)