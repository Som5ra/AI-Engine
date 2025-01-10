#include "csrt3d.h"
#include <filesystem>
#include <chrono>
#include <opencv2/opencv.hpp>

extern "C"{

    int GustoModelTargetInit(GustoModelTarget** model_target_ptr, const int height, const int width)
    {
        GustoModelTarget* model_target = new GustoModelTarget(height, width);
        *model_target_ptr = model_target;
        return 0;
    }

    int CADModelInit(GustoModelTarget* model_target_ptr, // ptr
        const char* model_name, const char* model_path, const char* model_metadata_path, // model_info,
        float start_threshold,
        float track_threshold,
        float* init_pose_ret_ptr // init_pose for unity rendering
    ){
        //setup CAD model
        const float unit_in_meter = 2.0;
        const float sphere_radius = 0.8;
        const bool from_opengl = false;
        const Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
        const float threshold_on_init = start_threshold;
        const float threshold_on_track = track_threshold;
        const float kl_threshold = 1.0;
        const bool debug_visualize = false;



        model_target_ptr->model_ptr_ = std::make_shared<RegionModel>(
            std::string(model_name),
            std::string(model_path),
            std::string(model_metadata_path),
            unit_in_meter,
            sphere_radius,
            from_opengl,
            transform,
            threshold_on_init,
            threshold_on_track,
            kl_threshold,
            debug_visualize
        );

        model_target_ptr->model_ptr_->Setup();

        model_target_ptr->init_pose = (Eigen::Matrix4f() << 
            -1, 0, 0, 0,
            0, 0, -1, 0,
            0, -1, 0, 0.8,
            0, 0, 0, 1
        ).finished();
        model_target_ptr->model_ptr_->reset_pose(model_target_ptr->init_pose);
        for(int i = 0; i < 16; i++){
            init_pose_ret_ptr[i] = model_target_ptr->init_pose.data()[i];
        }
        return 0;
    }

    int TrackerInit(GustoModelTarget* model_target_ptr, const float fov
    ){
        const float fx = (model_target_ptr->width / 2) / std::tan(3.1415 * fov / 360.0);
        // const float fx = 500.0;
        const float fy = fx;
        const float cx = model_target_ptr->width / 2;
        const float cy = model_target_ptr->height / 2;
        const Eigen::Matrix3f K = (Eigen::Matrix3f() << fx, 0, cx, 0, fy, cy, 0, 0, 1).finished();
        // setup tracker

        const int corr_iter = 7;
        const int pose_iter = 2;
        model_target_ptr->tracker_ptr_ = std::make_shared<RegionTracker>(model_target_ptr->width, model_target_ptr->height, K, fx, fy, cx, cy, corr_iter, pose_iter);
        model_target_ptr->tracker_ptr_->add_model(model_target_ptr->model_ptr_);
        model_target_ptr->tracker_ptr_->setup();
        return 0;
    }

    int reinit(GustoModelTarget* model_target_ptr
    ){
        model_target_ptr->model_ptr_->reset_pose(model_target_ptr->init_pose);
        return 0;
    }

    int track(GustoModelTarget* model_target_ptr,
        // unsigned char* bitmap,
        char* bitmap,
        float* pose_ret_ptr,
        float* conf_ret_ptr
    ){
  
        cv::Mat frame = cv::Mat(model_target_ptr->height, model_target_ptr->width, CV_8UC4, bitmap);
        cv::cvtColor(frame, frame, cv::COLOR_RGBA2RGB);
        cv::flip(frame, frame, 0);
        cv::resize(frame, frame, cv::Size(model_target_ptr->width, model_target_ptr->height));

        model_target_ptr->tracker_ptr_->track(frame, std::nullopt);
        conf_ret_ptr[0] = model_target_ptr->model_ptr_->conf_;
        for(size_t i = 0; i < 16; i++){
            pose_ret_ptr[i] = model_target_ptr->model_ptr_->pose().data()[i];
        }
        return 0;
    }



}