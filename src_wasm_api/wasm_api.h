
#include "utils.h"
#include "two_stage_human_pose_extractor_2d.h"
#include "multi_stage_face_geometry_3d.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
// #ifdef __EMSCRIPTEN__
// #else
// #define EMSCRIPTEN_KEEPALIVE
// #endif

#include <emscripten/emscripten.h>
#include <emscripten/bind.h>

using namespace emscripten;
