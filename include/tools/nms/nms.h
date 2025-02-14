#ifndef GUSTO_NMS_H
#define GUSTO_NMS_H
#include "utils.h"

namespace gusto_nms {


std::vector<int> nms_cpu(const std::vector<GustoRect>& _boxes, const std::vector<float>& _scores, float _score_thr, float _nms_thr);

std::pair<std::vector<int>, std::vector<int>> multiclass_nms_class_unaware_cpu(const std::vector<GustoRect>& boxes, const std::vector<std::vector<float>>& scores, float score_thr, float nms_thr);

} // namespace gusto_nms
#endif