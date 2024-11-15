#include "tools/face_geometry/geometry_pipeline.h"
#include "tools/face_geometry/face_geometry.h"
#include "tools/face_geometry/procrustes_solver.h"
#include "tools/face_geometry/calculator.h"
#include "tools/nms/nms.h"
#include "utils.h"
#include "face_model_family.h"

#include <onnxruntime_cxx_api.h>

#include <tuple>
#include <map>
#include <cmath>

#if defined(BUILD_PLATFORM_LINUX)
#include <opencv2/opencv.hpp>
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif


cv::Mat draw_axis(cv::Mat img, cv::Mat R, cv::Mat t, cv::Mat K, cv::Point2f origin) {
    // unit is mm
    cv::Mat rotV;
    cv::Rodrigues(R, rotV);
    std::vector<cv::Point3f> points = { {10, 0, 0}, {0, 10, 0}, {0, 0, 10}, {0, 0, 0} };
    std::vector<cv::Point2f> axisPoints;
    cv::projectPoints(points, rotV, t, K, cv::Mat(), axisPoints);

    for (auto& point : axisPoints) {
        point.x = img.cols - point.x;
    }

    // Compute direction vectors
    cv::Point2f x_axis = axisPoints[0] - axisPoints[3];
    cv::Point2f y_axis = axisPoints[1] - axisPoints[3];
    cv::Point2f z_axis = axisPoints[2] - axisPoints[3];

    // Use origin to draw the axes
    std::cout << "Origin: " << origin << std::endl;
    cv::circle(img, origin, 5, cv::Scalar(255, 255, 255), -1);
    cv::line(img, origin, origin + x_axis, cv::Scalar(255, 0, 0), 3); // X-axis in red
    cv::line(img, origin, origin + y_axis, cv::Scalar(0, 255, 0), 3); // Y-axis in green
    cv::line(img, origin, origin + z_axis, cv::Scalar(0, 0, 255), 3); // Z-axis in blue

    return img;
}


int main(int argc, char *argv[])
{

    bool DISPLAY = true;
    if (argc == 2 && argv[1] == std::string("no_display")) {
        DISPLAY = false;
    }

    std::string face_detector_path = "face_detector.onnx";
    std::string anchor_path = "anchor.bin";
    std::string face_landmarker_path = "face_landmarks_detector.onnx";
    std::cout << "Loading Face Detector Model: " << face_detector_path << std::endl;
    FaceDetector face_detector(face_detector_path, anchor_path);
    std::cout << "Loading Face Landmarker Model: " << face_landmarker_path << std::endl;
    FaceLandmarker face_landmarker(face_landmarker_path);

    // face_detector.check_names();


    gusto_face_geometry::FaceMeshCalculator face_mesh_calculator;
    const std::string face_GeometryPipelineMetadata = "geometry_pipeline_metadata_including_iris_landmarks.json";
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
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 720);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.open(0);
    }catch (const std::exception& e) {
        std::cerr << "Failed to open camera! Load Same Image Instead" << std::endl;
    }
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera! Load Same Image Instead" << std::endl;
    }
    if (DISPLAY){
        cv::namedWindow("Frame", cv::WINDOW_NORMAL);
    }
    #endif

    cv::Mat frame;

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
            // cv::Mat cropped_face = face_landmarker.crop_face(frame, box_to_crop);
            auto [cropped_face, box_to_crop_with_margin]= face_landmarker.crop_face(frame, box_to_crop);
            auto [points, tongueOut, score] = face_landmarker.forward(cropped_face);
            if (score < 0.49) {
                continue;
            }
            gusto_face_geometry::NormalizedLandmarkList thislandmark;
            for (auto pt : points) {
                gusto_face_geometry::NormalizedLandmark landmark;
                landmark.x = (pt.x + box_to_crop_with_margin[1]) / frame.size[1];
                landmark.y = (pt.y + box_to_crop_with_margin[0])/ frame.size[0];
                // landmark.x = (pt.x + box_to_crop_with_margin[0]) / frame.size[0];
                // landmark.y = (pt.y + box_to_crop_with_margin[1])/ frame.size[1];
                // landmark.z = pt.z / 500;
                landmark.z = pt.z / frame.size[1];
                thislandmark.landmark.push_back(landmark);
            }
            multi_face_landmarks.push_back(thislandmark);
            if (DISPLAY){
                // cropped_face = face_landmarker.draw_points(cropped_face, points);
                face_landmarker.draw_points(frame, points, cv::Point(box_to_crop_with_margin[1], box_to_crop_with_margin[0]));
                // cv::imshow("cropped_face", cropped_face);
            }
        }
        auto [multi_pose_mat, process_status] = face_mesh_calculator.Process(std::make_pair(frame.size[0], frame.size[1]), multi_face_landmarks);

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);

        K.at<float>(0, 0) = 800; // Focal length in x direction
        K.at<float>(1, 1) = 800; // Focal length in y direction
        K.at<float>(0, 2) = 320; // Principal point x-coordinate
        K.at<float>(1, 2) = 240; // Principal point y-coordinate
        if (process_status == GustoStatus::ERR_OK) {
            // std::cout << "Face Geometry Processed!" << std::endl;
            // for (auto fg : multi_pose_mat){
            for (size_t idx = 0; idx < multi_pose_mat.size(); idx++){
                cv::Mat R = cv::Mat(3, 3, CV_32F);
                cv::Mat t = cv::Mat(3, 1, CV_32F);
                for (int i = 0; i < 4; ++i) {
                    std::cout << std::endl;
                    for (int j = 0; j < 4; ++j) {
                        std::cout << multi_pose_mat[idx].pose_transform_matrix.at(i, j) << " ";
                        if (i < 3 && j < 3){
                            R.at<float>(i, j) = multi_pose_mat[idx].pose_transform_matrix.at(i, j);
                        }
                        if (i < 3 && j == 3){
                            t.at<float>(i, 0) = multi_pose_mat[idx].pose_transform_matrix.at(i, j);
                        }
                    }
                    std::cout << std::endl;
                }
                if (DISPLAY){
                    cv::Point2d nose = cv::Point2d(multi_face_landmarks[idx].landmark[4].x * frame.size[1], multi_face_landmarks[idx].landmark[4].y * frame.size[0]);
                    frame = draw_axis(frame, R, t, K, nose);
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