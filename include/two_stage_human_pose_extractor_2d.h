#ifndef TWO_STAGE_HUMAN_POSE_TRACKER_2D_H
#define TWO_STAGE_HUMAN_POSE_TRACKER_2D_H

#include "utils.h"
#include "detector2d_family.h"
#include "human_pose_family.h"
#include <onnxruntime_cxx_api.h>

#if defined(BUILD_PLATFORM_LINUX) && defined(DEBUG)
#include <opencv2/opencv.hpp>
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

#define GUSTO_RET int


class HumanPoseExtractor2D {
public:
    HumanPoseExtractor2D(const std::string& human_detector_model_path,
                       const std::string& human_detector_config_path,
                       const std::string& pose_detector_model_path,
                       const std::string& pose_detector_config_path,
                       int detect_interval = 3);
    ~HumanPoseExtractor2D();

    GUSTO_RET DetectPose(const cv::Mat& image);
    GUSTO_RET Display(cv::Mat& image, bool display_box = true, bool display_keypoints = true);
    GUSTO_RET Debug();
private:
    std::unique_ptr<gusto_detector2d::Detector> human_detector;
    std::unique_ptr<gusto_humanpose::RTMPose> pose_detector;
    // std::unique_ptr<gusto_detector2d::DetectionResult> detection_result;
    // std::unique_ptr<gusto_humanpose::KeyPoint2DResult> pose_result;
    gusto_detector2d::DetectionResult* detection_result;
    gusto_humanpose::KeyPoint2DResult* pose_result;
    
    int detect_interval;
    int num_frames;
    
    float scale;
    float LetterBoxImage(
        const cv::Mat& image,
        cv::Mat& out_image,
        const cv::Size& new_shape = cv::Size(640, 640),
        int stride = 32,
        const cv::Scalar& color = cv::Scalar(114, 114, 114),
        bool fixed_shape = false,
        bool scale_up = true);
    
    
};

#endif // TWO_STAGE_HUMAN_POSE_TRACKER_2D_H