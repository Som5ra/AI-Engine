#include "two_stage_human_pose_extractor_2d.h"

#include <algorithm>
#include <cmath>
#include <iostream>

float HumanPoseExtractor2D::LetterBoxImage(
    const cv::Mat& image,
    cv::Mat& out_image,
    const cv::Size& new_shape,
    int stride,
    const cv::Scalar& color,
    bool fixed_shape,
    bool scale_up) const {
    const cv::Size shape = image.size();
    float resize_ratio = std::min(
        static_cast<float>(new_shape.height) / shape.height,
        static_cast<float>(new_shape.width) / shape.width);

    if (!scale_up) {
        resize_ratio = std::min(resize_ratio, 1.0F);
    }

    const cv::Size resized_shape{
        static_cast<int>(std::round(shape.width * resize_ratio)),
        static_cast<int>(std::round(shape.height * resize_ratio)),
    };

    cv::Mat resized_image;
    if (shape != resized_shape) {
        cv::resize(image, resized_image, resized_shape);
    } else {
        resized_image = image.clone();
    }

    int horizontal_padding = new_shape.width - resized_shape.width;
    int vertical_padding = new_shape.height - resized_shape.height;
    if (!fixed_shape) {
        horizontal_padding %= stride;
        vertical_padding %= stride;
    }

    cv::copyMakeBorder(
        resized_image,
        out_image,
        0,
        vertical_padding,
        0,
        horizontal_padding,
        cv::BORDER_CONSTANT,
        color);

    return 1.0F / resize_ratio;
}

HumanPoseExtractor2D::HumanPoseExtractor2D(
    const std::string& human_detector_model_path,
    const std::string& human_detector_config_path,
    const std::string& pose_detector_model_path,
    const std::string& pose_detector_config_path,
    int detect_interval)
    : human_detector_(std::make_unique<gusto_detector2d::Detector>(
          human_detector_model_path, human_detector_config_path)),
      pose_detector_(std::make_unique<gusto_humanpose::RTMPose>(
          pose_detector_model_path, pose_detector_config_path)),
      detection_result_(
          std::make_unique<gusto_detector2d::DetectionResult>()),
      detect_interval_(std::max(1, detect_interval)) {}

GUSTO_RET HumanPoseExtractor2D::DetectPose(const cv::Mat& image) {
    if (image.empty() || !human_detector_ || !pose_detector_ ||
        !detection_result_) {
        return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    try {
        cv::Mat detector_frame;
        display_scale_ = LetterBoxImage(
            image,
            detector_frame,
            cv::Size(320, 320),
            32,
            cv::Scalar(128, 128, 128),
            true);

        const bool refresh_detection = frame_index_ == 0;
        frame_index_ = (frame_index_ + 1) % detect_interval_;

        if (refresh_detection) {
            auto detector_output = human_detector_->forward(detector_frame);
            auto* detector_result =
                dynamic_cast<gusto_detector2d::DetectionResult*>(
                    detector_output.get());
            if (detector_result == nullptr) {
                return GustoStatus::ERR_GENERAL_ERROR;
            }
            detection_result_->boxes = detector_result->boxes;
        }

        pose_results_.clear();
        pose_results_.resize(detection_result_->boxes.size());

        for (std::size_t index = 0;
             index < detection_result_->boxes.size();
             ++index) {
            auto [pose_input, inverse_affine_transform] =
                pose_detector_->CropImageByDetectBox(
                    detector_frame, detection_result_->boxes[index]);
            (void)inverse_affine_transform;

            if (pose_input.empty()) {
                continue;
            }

            auto pose_output = pose_detector_->forward(pose_input);
            auto* pose_result =
                dynamic_cast<gusto_humanpose::KeyPoint2DResult*>(
                    pose_output.get());
            if (pose_result == nullptr) {
                return GustoStatus::ERR_GENERAL_ERROR;
            }

            pose_results_[index].keypoints = pose_result->keypoints;
        }

        return GustoStatus::ERR_OK;
    } catch (const cv::Exception& exception) {
        std::cerr << "Human pose OpenCV error: " << exception.what()
                  << std::endl;
    } catch (const std::exception& exception) {
        std::cerr << "Human pose inference error: " << exception.what()
                  << std::endl;
    }

    pose_results_.clear();
    return GustoStatus::ERR_GENERAL_ERROR;
}

GUSTO_RET HumanPoseExtractor2D::Display(
    cv::Mat& image,
    bool display_box,
    bool display_keypoints) {
    if (image.empty() || !detection_result_ || !pose_detector_) {
        return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    const auto& boxes = detection_result_->boxes;
    const std::size_t result_count = std::min(boxes.size(), pose_results_.size());

    for (std::size_t index = 0; index < result_count; ++index) {
        if (display_box) {
            cv::rectangle(
                image,
                cv::Point(
                    static_cast<int>(boxes[index].x1 * display_scale_),
                    static_cast<int>(boxes[index].y1 * display_scale_)),
                cv::Point(
                    static_cast<int>(boxes[index].x2 * display_scale_),
                    static_cast<int>(boxes[index].y2 * display_scale_)),
                cv::Scalar(0, 255, 0),
                2);
        }

        if (display_keypoints && !pose_results_[index].keypoints.empty()) {
            image = pose_detector_->draw_single_person_keypoints(
                image,
                pose_results_[index].keypoints,
                display_scale_);
        }
    }

    return GustoStatus::ERR_OK;
}

GUSTO_RET HumanPoseExtractor2D::Debug() const {
    if (!human_detector_ || !human_detector_->_config) {
        return GustoStatus::ERR_GENERAL_ERROR;
    }

    std::cout << human_detector_->_config->model_path << std::endl;
    return GustoStatus::ERR_OK;
}
