#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>
#include <opencv2/opencv.hpp> 

using namespace std;
using namespace cv;
using namespace cv::dnn;



// pair<vector<int>, vector<int>> multiclass_nms_class_unaware_cpu(const vector<Rect>& boxes, const Mat& scores, float score_thr, float nms_thr) {
//     vector<int> cls_inds(scores.rows);
//     vector<float> cls_scores(scores.rows);
//     for (int i = 0; i < scores.rows; ++i) {
//         Point maxLoc;
//         minMaxLoc(scores.row(i), nullptr, &cls_scores[i], nullptr, &maxLoc);
//         cls_inds[i] = maxLoc.x;
//     }

//     vector<int> valid_idx;
//     vector<int> valid_idx_class_id;
//     vector<int> indices;
//     for (int i = 0; i < cls_scores.size(); ++i) {
//         if (cls_scores[i] >= score_thr) {
//             indices.push_back(i);
//         }
//     }

//     vector<Rect> filtered_boxes;
//     vector<float> filtered_scores;
//     for (int idx : indices) {
//         filtered_boxes.push_back(boxes[idx]);
//         filtered_scores.push_back(cls_scores[idx]);
//     }

//     vector<int> nms_indices;
//     dnn::NMSBoxes(filtered_boxes, filtered_scores, score_thr, nms_thr, nms_indices);

//     for (int idx : nms_indices) {
//         valid_idx.push_back(indices[idx]);
//         valid_idx_class_id.push_back(cls_inds[indices[idx]]);
//     }

//     return {valid_idx, valid_idx_class_id};
// }

// int main()
// {
//     const int input_size = 640;
//     cout << "available threads: " << omp_get_num_procs() << endl;
//     cv::setNumThreads(std::max(1, omp_get_num_procs() / 2));
//     string onnx_model = "/media/sombrali/HDD1/mmlib/mmyolo/work_dirs/rtmdet_tiny_fast_1xb12-40e_cat/" + to_string(input_size) + "x" + to_string(input_size) + "/end2end_nonms_fp16.onnx";
//     // string onnx_model = "/media/sombrali/HDD1/mmlib/mmyolo/work_dirs/rtmdet_tiny_fast_1xb12-40e_cat/320x320/end2end_nonms.onnx";
//     Net net = readNetFromONNX(onnx_model);
//     net.setPreferableBackend(DNN_BACKEND_OPENCV);
//     net.setPreferableTarget(DNN_TARGET_CPU);

//     // using opencv to call webcam
//     cv::VideoCapture cap(1);
//     if (!cap.isOpened()) {
//         std::cerr << "Error opening video stream or file" << std::endl;
//         return -1;
//     }


//     float fps = 0.0;
//     while (true)
//     {
//         auto start = chrono::high_resolution_clock::now();
//         Mat raw;
//         cap >> raw;
//         if (raw.empty()) {
//             cerr << "Error: Captured empty frame." << endl;
//             break;
//         }
//         // cout << "Captured frame size: " << frame.size() << endl;

//         // Preprocess the frame
//         Mat inputBlob;
//         cvtColor(raw, inputBlob, COLOR_BGR2RGB);
//         resize(inputBlob, inputBlob, Size(input_size, input_size));
//         inputBlob.convertTo(inputBlob, CV_32F, 1.0 / 255);
//         auto pimage_time = chrono::high_resolution_clock::now();
//         // Run inference
//         try {
//             net.setInput(blobFromImage(inputBlob));
//             // auto setinputtime = chrono::high_resolution_clock::now();
//             // cout << "net.forward " << chrono::duration_cast<chrono::milliseconds>(setinputtime - pimage_time).count() << "ms" << endl;

//             vector<Mat> outs;
//             net.forward(outs, net.getUnconnectedOutLayersNames());
//             // auto forward_time = chrono::high_resolution_clock::now();
//             // cout << "net.forward " << chrono::duration_cast<chrono::milliseconds>(forward_time - setinputtime).count() << "ms" << endl;

//             // Process output
//             if (outs.size() < 2) {
//                 cerr << outs.size()  << endl;
//                 cerr << outs[0].size()  << endl;
//                 cerr << "Error: Unexpected number of outputs from the model." << endl;
//                 break;
//             }


//             Mat boxes = outs[0].reshape(1, {outs[0].size[1], outs[0].size[2]});
//             Mat scores = outs[1].reshape(1, {outs[1].size[1], outs[1].size[2]});

//             vector<Rect> box_list;
//             for (int i = 0; i < boxes.rows; ++i) {
//                 box_list.push_back(Rect(Point(boxes.at<float>(i, 0), boxes.at<float>(i, 1)),
//                                         Point(boxes.at<float>(i, 2), boxes.at<float>(i, 3))));
//             }
//             auto decode_time = chrono::high_resolution_clock::now();
//             // cout << "decode " << chrono::duration_cast<chrono::milliseconds>(decode_time - forward_time).count() << "ms" << endl;

//             cv::putText(raw, "FPS: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
//             // Display the frame
//             imshow("Frame", raw);
//             if (waitKey(10) == 27) // Press 'ESC' to exit
//                 break;
//         } catch (const cv::Exception& e) {
//             cerr << "Error during inference: " << e.what() << endl;
//             break;
//         }

//         auto end = chrono::high_resolution_clock::now();
//         // cout << "forward time: " << chrono::duration_cast<chrono::milliseconds>(end - pimage_time).count() << "ms" << endl;

//         // cout << "Inference time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
//         fps = 1000.0 / chrono::duration_cast<chrono::milliseconds>(end - start).count();
//     }
//     // vector<Rect> box_list;
//     // for (int i = 0; i < boxes.rows; ++i) {
//     //     box_list.push_back(Rect(Point(boxes.at<float>(i, 0), boxes.at<float>(i, 1)),
//     //                             Point(boxes.at<float>(i, 2), boxes.at<float>(i, 3))));
//     // }

//     // auto [indices, indices_cls] = multiclass_nms_class_unaware_cpu(box_list, scores, 0.4, 0.5);

//     // // Rescale to original size and draw boxes
//     // for (size_t i = 0; i < indices.size(); ++i) {
//     //     Rect box = box_list[indices[i]];
//     //     box.x *= w_ratio;
//     //     box.y *= h_ratio;
//     //     box.width *= w_ratio;
//     //     box.height *= h_ratio;
//     //     rectangle(raw, box, Scalar(0, 255, 0), 2);
//     //     putText(raw, to_string(scores.at<float>(indices[i], indices_cls[i])), Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
//     // }

//     // // Display image
//     // imshow("image", raw);
//     // waitKey(0);
//     // destroyAllWindows();

//     return 0;
// }


#include "detector2d.h"
#include <iostream>
#include <vector>
int main()
{
    GenericDetector* detector = new GenericDetector();

    // const std::string modelpath = "/media/sombrali/HDD1/opencv-unity/gusto_dnn/weights/end2end_nonms_fp16.onnx";
    const std::string modelpath = "/media/sombrali/HDD1/mmlib/mmyolo/work_dirs/retinanet_mbnv2-1x_coco/epoch_12/end2end_nonms.onnx";
    const std::string cls_names_path = "/media/sombrali/HDD1/opencv-unity/gusto_dnn/weights/rtm_test_cat.names";
    std::cout << "modelpath: " << modelpath << std::endl;
    int _compile_error_code = detector->compile(640, 640, 0.5, 0.5, modelpath.c_str(), cls_names_path.c_str(), 1024);
    std::cout << "compile error code: " << _compile_error_code << std::endl;    

    // using opencv to call webcam
    cv::VideoCapture cap(1);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }

    float* ret_bboxes = new float[1000];
    float* ret_confidences = new float[1000];
    int* ret_classIds = new int[1000];
    int* ret_len = new int;

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        detector->detect(frame, ret_bboxes, ret_confidences, ret_classIds, ret_len);
        if (frame.empty())
            break;
        cv::imshow("Frame", frame);
        if (cv::waitKey(10) == 27)
            break;
    }
    

    return 0;
}