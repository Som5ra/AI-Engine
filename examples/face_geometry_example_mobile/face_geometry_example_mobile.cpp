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


extern "C"{


    float min_time = 1000000;
    float max_time = 0;
    float total_time = 0;
    int num_frames = 0;
    
    FaceDetector* face_detector;
    FaceLandmarker* face_landmarker;
    gusto_face_geometry::FaceMeshCalculator face_mesh_calculator;


    cv::Mat frame;

    void Open_Session(
        char* _face_detector_path, 
        char* _face_landmarker_path, 
        char* _face_GeometryPipelineMetadata,
        char* _anchor_path
    ){
        std::string face_detector_path;
        std::string face_landmarker_path;
        std::string face_GeometryPipelineMetadata;
        std::string anchor_path;
        face_detector_path.assign(_face_detector_path);
        face_landmarker_path.assign(_face_landmarker_path);
        face_GeometryPipelineMetadata.assign(_face_GeometryPipelineMetadata);
        anchor_path.assign(_anchor_path);

        std::cout << "Loading Face Detector Model: " << face_detector_path << std::endl;
        face_detector = new FaceDetector(face_detector_path, anchor_path);
        std::cout << "Loading Face Landmarker Model: " << face_landmarker_path << std::endl;
        face_landmarker = new FaceLandmarker(face_landmarker_path);    
    
        GUSTO_RET open_status = face_mesh_calculator.Open(face_GeometryPipelineMetadata);
        if (open_status != GustoStatus::ERR_OK) {
            std::cerr << "Failed to open Geometry Pipeline Metadata!" << std::endl;
            return ;
        }
    }


    void Start_Session(char* _frame_path){
        std::string frame_path;
        frame_path.assign(_frame_path);
        std::cout << "Processing Frame: " << frame_path << std::endl;
        frame = cv::imread(frame_path);
        auto start = std::chrono::high_resolution_clock::now();
        auto [boxes, scores, indices, indices_cls] = face_detector->forward(frame);
        std::vector<gusto_face_geometry::NormalizedLandmarkList> multi_face_landmarks;
        for(size_t idx = 0; idx < indices.size(); idx++) {
            std::vector<int> box_to_crop = {
                static_cast<int>(boxes[indices[idx]].y1 * frame.size[0]),
                static_cast<int>(boxes[indices[idx]].x1 * frame.size[1]),
                static_cast<int>(boxes[indices[idx]].y2 * frame.size[0]),
                static_cast<int>(boxes[indices[idx]].x2 * frame.size[1]), 
            }; 
            // cv::Mat cropped_face = face_landmarker.crop_face(frame, box_to_crop);
            auto [cropped_face, box_to_crop_with_margin]= face_landmarker->crop_face(frame, box_to_crop);
            auto [points, tongueOut, score] = face_landmarker->forward(cropped_face);
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
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        min_time = std::min(min_time, duration);
        max_time = std::max(max_time, duration);
        total_time += duration;
        num_frames++;
        std::cout << "\rmin_time: " << min_time << "ms  |  max_time: " << max_time << "ms  |  avg_time: " << std::ceil(total_time / num_frames * 100) / 100 << "ms " << std::flush;    
        return ;
    }

}