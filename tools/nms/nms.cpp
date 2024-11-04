#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
#include <omp.h>
#endif

using namespace std;



struct Rect {
    float x1, y1, x2, y2;
    Rect(float x1, float y1, float x2, float y2) :  x1(x1), y1(y1), x2(x2), y2(y2) {}
    float area() const { return (y2 - y1 + 1) * (x2 - x1 + 1); } 
};

vector<int> nms_cpu(const vector<Rect>& _boxes, const vector<float>& _scores, float _score_thr, float _nms_thr) {
    /*
    Single class NMS
    
    inputs:
        boxes: NDArray (num_boxes, 4) in xyxy
        scores: NDArray (num_boxes, 1) in [0,1]
    
    output:
        NDArray of indices to keep
    */
    vector<int> _raw_indices(_scores.size()), raw_indices;
    vector<Rect> boxes;
    vector<float> scores;
    vector<float> areas; 

    iota(_raw_indices.begin(), _raw_indices.end(), 0);
    for (size_t i = 0; i < _scores.size(); ++i) {
        if (_scores[i] >= _score_thr) {
            boxes.push_back(_boxes[i]);
            scores.push_back(_scores[i]);
            raw_indices.push_back(_raw_indices[i]);
            areas.push_back(_boxes[i].area()); 
        }
    }
    vector<int> order(scores.size());
    iota(order.begin(), order.end(), 0);
    stable_sort(order.begin(), order.end(), [&scores](size_t i1, size_t i2) {return scores[i1] > scores[i2];});
    for (size_t i = 0; i < order.size(); i++) {
        cerr << order[i] << " " << scores[order[i]] << " " << raw_indices[order[i]] << " ";
    }
    cerr << endl;
    
    vector<int> keep;
    while (!order.empty()) {
        int i = order[0];
        keep.push_back(raw_indices[i]);
        vector<int> new_order;
        for (size_t j = 1; j < order.size(); ++j) {
            int idx = order[j];
            float xx1 = max(boxes[i].x1, boxes[idx].x1);
            float yy1 = max(boxes[i].y1, boxes[idx].y1);
            float xx2 = min(boxes[i].x2, boxes[idx].x2);
            float yy2 = min(boxes[i].y2, boxes[idx].y2);

            float w = max(0.0f, xx2 - xx1 + 1);
            float h = max(0.0f, yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (areas[i] + areas[idx] - inter);
            if (ovr <= _nms_thr) {
                new_order.push_back(idx);
            }
        }
        order = new_order;
    }

    return keep;
}

pair<vector<int>, vector<int>> multiclass_nms_class_unaware_cpu(const vector<Rect>& boxes, const vector<vector<float>>& scores, float score_thr, float nms_thr) {
    /*
    Mutli class NMS (class-unaware)

    Class-unaware: a proposal can only belong to a single class
    
    inputs:
        boxes: NDArray (num_boxes, 4) in xyxy
        scores: NDArray (num_boxes, num_classes) in [0, 1]
    
    output:
        [NDArray of indices to keep, NDArray of class id]
    */
    
    vector<int> cls_inds(scores.size());
    vector<float> cls_scores(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        auto max_it = max_element(scores[i].begin(), scores[i].end());
        cls_inds[i] = distance(scores[i].begin(), max_it);
        cls_scores[i] = *max_it;
    }
    vector<int> valid_idx = nms_cpu(boxes, cls_scores, score_thr, nms_thr);
    for (size_t i = 0; i < valid_idx.size(); ++i) {
        cerr << valid_idx[i] << " ";
    }
    if (valid_idx.empty()) {
        return {vector<int>(), vector<int>()};
    }

    vector<int> valid_idx_class_id(valid_idx.size());
    for (size_t i = 0; i < valid_idx.size(); ++i) {
        valid_idx_class_id[i] = cls_inds[valid_idx[i]];
    }

    return {valid_idx, valid_idx_class_id};
}

void sigmoid(float* arr, int size) {
    #if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
    omp_set_num_threads(omp_get_max_threads() / 2);
    #pragma omp parallel for
    #endif
    for (int i = 0; i < size; i++) {
        arr[i] = 1 / (1 + exp(-arr[i]));
    }
}

extern "C" {

    void nms_with_sigmoid(float* boxes, int* boxes_shape, float* scores, int* scores_shape, float score_thr, float nms_thr, int* ret_indices, int* ret_indices_cls, int* ret_len) {
        /*
            For example,
            boxes_shape: (batch_size, 2100, 4)
            scores_shape: (batch_size, 3, 2100)

            ret_indices: (batch_size, n,) 
            ret_indices_cls: (batch_size, n,) 
            ret_len: (batch_size, 1), where n is the number of valid boxes so we can decode the boxes in c# side
        */
        int boxarr_size_per_batch = boxes_shape[1] * boxes_shape[2];
        int scorearr_size_per_batch = scores_shape[1] * scores_shape[2];
        int current_ret_len = 0;   
        sigmoid(scores, scores_shape[0] * scores_shape[1] * scores_shape[2]);

        #if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
        omp_set_num_threads(omp_get_max_threads() / 2);
        #pragma omp parallel for
        #endif
        for (size_t batch = 0; batch < boxes_shape[0]; batch++) {
            int box_batch_offset = batch * boxarr_size_per_batch;
            int score_batch_offset = batch * scorearr_size_per_batch;
            vector<Rect> boxes_vec;
            vector<vector<float>> scores_vec;
            for (int i = 0; i < boxes_shape[1]; i++) {
                boxes_vec.push_back(Rect(boxes[box_batch_offset + i * 4], boxes[box_batch_offset + i * 4 + 1], boxes[box_batch_offset + i * 4 + 2], boxes[box_batch_offset + i * 4 + 3]));
            }
            for (int i = 0; i < scores_shape[2]; i++) {
                vector<float> scores_vec_tmp;
                for (int j = 0; j < scores_shape[1]; j++) {
                    scores_vec_tmp.push_back(scores[score_batch_offset + j * scores_shape[2] + i]);
                }
                scores_vec.push_back(scores_vec_tmp);
            }
            pair<vector<int>, vector<int>> res = multiclass_nms_class_unaware_cpu(boxes_vec, scores_vec, score_thr, nms_thr);
            ret_len[batch] = res.first.size();
            for (size_t i = 0; i < res.first.size(); i++) {
                ret_indices[current_ret_len + i] = res.first[i];
                ret_indices_cls[current_ret_len + i] = res.second[i];
            }
            current_ret_len += res.first.size();
        }
        return;
    }

    void nms(float* boxes, int* boxes_shape, float* scores, int* scores_shape, float score_thr, float nms_thr, int* ret_indices, int* ret_indices_cls, int* ret_len) {
        /*
            For example,
            boxes_shape: (batch_size, 2100, 4)
            scores_shape: (batch_size, 3, 2100)

            ret_indices: (batch_size, n,) 
            ret_indices_cls: (batch_size, n,) 
            ret_len: (batch_size, 1), where n is the number of valid boxes so we can decode the boxes in c# side
        */
        int boxarr_size_per_batch = boxes_shape[1] * boxes_shape[2];
        int scorearr_size_per_batch = scores_shape[1] * scores_shape[2];
        int current_ret_len = 0;   
        #if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
        omp_set_num_threads(omp_get_max_threads() / 2);
        #pragma omp parallel for
        #endif
        for (size_t batch = 0; batch < boxes_shape[0]; batch++) {
            int box_batch_offset = batch * boxarr_size_per_batch;
            int score_batch_offset = batch * scorearr_size_per_batch;
            vector<Rect> boxes_vec;
            vector<vector<float>> scores_vec;
            for (int i = 0; i < boxes_shape[1]; i++) {
                boxes_vec.push_back(Rect(boxes[box_batch_offset + i * 4], boxes[box_batch_offset + i * 4 + 1], boxes[box_batch_offset + i * 4 + 2], boxes[box_batch_offset + i * 4 + 3]));
            }
            for (int i = 0; i < scores_shape[2]; i++) {
                vector<float> scores_vec_tmp;
                for (int j = 0; j < scores_shape[1]; j++) {
                    scores_vec_tmp.push_back(scores[score_batch_offset + j * scores_shape[2] + i]);
                }
                scores_vec.push_back(scores_vec_tmp);
            }
            pair<vector<int>, vector<int>> res = multiclass_nms_class_unaware_cpu(boxes_vec, scores_vec, score_thr, nms_thr);
            ret_len[batch] = res.first.size();
            for (size_t i = 0; i < res.first.size(); i++) {
                ret_indices[current_ret_len + i] = res.first[i];
                ret_indices_cls[current_ret_len + i] = res.second[i];
            }
            current_ret_len += res.first.size();
        }
        return;
    }
}

// int main() {
//     // Set the number of threads to half of the available threads
//     num_threads = omp_get_max_threads() / 2;
//    ;

//     vector<Rect> boxes = {Rect(0, 0, 10, 10), Rect(1, 1, 10, 10), Rect(2, 2, 10, 10),
//     Rect(0, 0, 10, 10), Rect(1, 1, 10, 10), Rect(2, 2, 10, 10),
//     Rect(0, 0, 10, 10), Rect(1, 1, 10, 10), Rect(2, 2, 10, 10),
//     Rect(0, 0, 10, 10), Rect(1, 1, 10, 10), Rect(2, 2, 10, 10),
//     Rect(0, 0, 10, 10), Rect(1, 1, 10, 10)};
//     vector<float> test1_scores = {
//         0.56884766, 0.7861328, 0.8598633, 0.5605469, 0.796875, 0.4099121,
//         0.80859375, 0.7529297, 0.6801758, 0.81591797, 0.7993164, 0.7807617,
//         0.7631836
//     };

//     vector<float> test2_scores = {
//         0.5800781, 0.7866211, 0.85546875, 0.5654297, 0.4428711, 0.7885742,
//         0.40893555, 0.80859375, 0.74902344, 0.6777344, 0.81689453, 0.7993164,
//         0.78222656
//     };
//     vector<vector<float>> scores1 = {test1_scores, test2_scores};
//     float score_thr = 0.5;
//     float nms_thr = 0.3;

//     auto [valid_idx, valid_idx_class_id] = multiclass_nms_class_unaware_cpu(boxes, scores1, score_thr, nms_thr);
//     cout << "Unaware NMS valid indices: ";
//     for (int idx : valid_idx) {
//         cout << idx << endl;
//         cout << boxes[idx].x1 << " " << boxes[idx].y1 << " " << boxes[idx].x2 << " " << boxes[idx].y2 << endl;
//         cout << endl;
//     }
//     cout << endl;

//     return 0;
// }