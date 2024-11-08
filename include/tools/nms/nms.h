#ifndef GUSTO_NMS_H
#define GUSTO_NMS_H
#include "utils.h"

namespace gusto_nms {
    
struct Rect {
    float x1, y1, x2, y2;
    Rect(float x1, float y1, float x2, float y2) :  x1(x1), y1(y1), x2(x2), y2(y2) {}
    float area() const { return (y2 - y1 + 1) * (x2 - x1 + 1); } 
};


std::vector<int> nms_cpu(const std::vector<Rect>& _boxes, const std::vector<float>& _scores, float _score_thr, float _nms_thr);

std::pair<std::vector<int>, std::vector<int>> multiclass_nms_class_unaware_cpu(const std::vector<Rect>& boxes, const std::vector<std::vector<float>>& scores, float score_thr, float nms_thr);

} // namespace gusto_nms
#endif