#include "tools/face_geometry/geometry_pipeline.h"
#include "tools/face_geometry/face_geometry.h"
#include "tools/face_geometry/procrustes_solver.h"
#include "tools/face_geometry/calculator.h"
#include "tools/nms/nms.h"
#include "utils.h"
#include "face_model_family.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <map>
#include <cmath>



int main(int argc, char *argv[])
{

    bool DISPLAY = true;
    if (argc == 2 && argv[1] == std::string("no_display")) {
        DISPLAY = false;
    }

    std::string face_detector_path = "/media/sombrali/HDD1/facelandmark/weights/mediapipe/face_detector.onnx";
    std::string face_landmarker_path = "/media/sombrali/HDD1/facelandmark/weights/mediapipe/face_landmarks_detector.onnx";

    FaceDetector face_detector(face_detector_path);
    FaceLandmarker face_landmarker(face_landmarker_path);

    // face_detector.check_names();


    gusto_face_geometry::FaceMeshCalculator face_mesh_calculator;
    const std::string face_GeometryPipelineMetadata = "/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/face_geometry/geometry_pipeline_metadata_including_iris_landmarks.json";
    GUSTO_RET open_status = face_mesh_calculator.Open(face_GeometryPipelineMetadata);
    if (open_status != GustoStatus::ERR_OK) {
        std::cerr << "Failed to open Geometry Pipeline Metadata!" << std::endl;
        return 1;
    }

    // cv::namedWindow("Frame", cv::WINDOW_NORMAL);
    // cv::Mat test_image = cv::imread("/media/sombrali/HDD1/opencv-unity/gusto_dnn/000000000785.jpg");
    // cv::Mat test_image = cv::imread("/media/sombrali/HDD1/opencv-unity/gusto_dnn/download.png", cv::IMREAD_COLOR);
    // auto [boxes, scores, indices, indices_cls] = face_detector.forward(test_image);
    // auto painted_image = face_detector.draw_boxes(test_image, boxes, scores, indices, indices_cls);
    // cv::imshow("Frame", painted_image);
    // cv::waitKey(0);



    float min_time = 1000000;
    float max_time = 0;
    float total_time = 0;
    int num_frames = 0;
    
    #if defined(BUILD_PLATFORM_LINUX)
    cv::VideoCapture cap;
    try{
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.open(0);
    }catch (const std::exception& e) {
        std::cerr << "Failed to open camera! Load Same Image Instead" << std::endl;
    }
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera! Load Same Image Instead" << std::endl;
    }
    #endif

    cv::Mat frame;
    if (DISPLAY){
        cv::namedWindow("Frame", cv::WINDOW_NORMAL);
    }
    // cv::namedWindow("cropped_face", cv::WINDOW_NORMAL);
    while (true)
    {
        #if defined(BUILD_PLATFORM_LINUX)
        if (!cap.isOpened()){
            frame = cv::imread("demo.png");            
        }else{
            cap >> frame; // so fkng slow
        }
        #else
        frame = cv::imread("demo.png");
        #endif
        auto start = std::chrono::high_resolution_clock::now();
        auto [boxes, scores, indices, indices_cls] = face_detector.forward(frame);
        std::vector<gusto_face_geometry::NormalizedLandmarkList> multi_face_landmarks;
        for(size_t idx = 0; idx < indices.size(); idx++) {
            std::vector<int> box_to_crop = {
                static_cast<int>(boxes[indices[idx]].y1 * frame.size[0]),
                static_cast<int>(boxes[indices[idx]].x1 * frame.size[1]),
                static_cast<int>(boxes[indices[idx]].y2 * frame.size[0]),
                static_cast<int>(boxes[indices[idx]].x2 * frame.size[1]), 
            }; 
            cv::Mat cropped_face = face_landmarker.crop_face(frame, box_to_crop);
            auto [points, tongueOut, score] = face_landmarker.forward(cropped_face);
            if (score < 0.49) {
                continue;
            }
            gusto_face_geometry::NormalizedLandmarkList thislandmark;
            for (auto pt : points) {
                gusto_face_geometry::NormalizedLandmark landmark;
                landmark.x = (pt.x + box_to_crop[1]) / frame.size[1];
                landmark.y = (pt.y + box_to_crop[0])/ frame.size[0];
                // landmark.x = (pt.x + box_to_crop[0]) / frame.size[0];
                // landmark.y = (pt.y + box_to_crop[1])/ frame.size[1];
                landmark.z = pt.z;
                thislandmark.landmark.push_back(landmark);
            }
            multi_face_landmarks.push_back(thislandmark);
            if (DISPLAY){
                cropped_face = face_landmarker.draw_points(cropped_face, points);
                // cv::imshow("cropped_face", cropped_face);
            }
        }
        auto [multi_pose_mat, process_status] = face_mesh_calculator.Process(std::make_pair(frame.size[0], frame.size[1]), multi_face_landmarks);
        if (process_status == GustoStatus::ERR_OK) {
            // std::cout << "Face Geometry Processed!" << std::endl;
            for (auto fg : multi_pose_mat){
                for (int i = 0; i < 4; ++i) {
                    std::cout << std::endl;
                    for (int j = 0; j < 4; ++j) {
                        std::cout << fg.pose_transform_matrix.at(i, j) << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        min_time = std::min(min_time, duration);
        max_time = std::max(max_time, duration);
        total_time += duration;
        num_frames++;
        std::cout << "\rmin_time: " << min_time << "ms  |  max_time: " << max_time << "ms  |  avg_time: " << std::ceil(total_time / num_frames * 100) / 100 << "ms " << std::flush;    

        if (DISPLAY){
            // auto painted_image = face_detector.draw_boxes(frame, boxes, scores, indices, indices_cls);
            cv::imshow("Frame", frame);
            if (cv::waitKey(25) >= 0)
                break;
        }


    }
    


    return 0;
}