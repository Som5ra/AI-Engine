#ifndef CUSTOM_NMS_H
#define CUSTOM_NMS_H
#include "utils.h"

namespace custom_nms {


std::vector<int> nms_cpu(const std::vector<CustomRect>& _boxes, const std::vector<float>& _scores, float _score_thr, float _nms_thr);

std::pair<std::vector<int>, std::vector<int>> multiclass_nms_class_unaware_cpu(const std::vector<CustomRect>& boxes, const std::vector<std::vector<float>>& scores, float score_thr, float nms_thr);

} // namespace custom_nms
#endif