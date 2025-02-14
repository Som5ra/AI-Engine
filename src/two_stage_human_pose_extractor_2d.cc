#include "two_stage_human_pose_extractor_2d.h"



float HumanPoseExtractor2D::LetterBoxImage(
	const cv::Mat& image,
	cv::Mat& out_image,
	const cv::Size& new_shape,
	int stride,
	const cv::Scalar& color,
	bool fixed_shape,
	bool scale_up) 
{
	cv::Size shape = image.size();
	float r = std::min((float)new_shape.height / (float)shape.height, (float)new_shape.width / (float)shape.width);

	if (!scale_up) {
		r = std::min(r, 1.0f);
	}

	int newUnpad[2]{
		(int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r) };

	cv::Mat tmp;
	if (shape.width != newUnpad[0] || shape.height != newUnpad[1]) {
		cv::resize(image, tmp, cv::Size(newUnpad[0], newUnpad[1]));
	}
	else {
		tmp = image.clone();
	}

	float dw = new_shape.width - newUnpad[0];
	float dh = new_shape.height - newUnpad[1];

	if (!fixed_shape) {
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}

	int top = int(0);
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(0));
	int right = int(std::round(dw + 0.1f));

	cv::copyMakeBorder(tmp, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);

	return 1.0f / r;
}


HumanPoseExtractor2D::HumanPoseExtractor2D(    
    const std::string& human_detector_model_path,
    const std::string& human_detector_config_path,
    const std::string& pose_detector_model_path,
    const std::string& pose_detector_config_path,
    int detect_interval) 
{
    this->human_detector = std::make_unique<gusto_detector2d::Detector>(human_detector_model_path, human_detector_config_path);
    this->pose_detector = std::make_unique<gusto_humanpose::RTMPose>(pose_detector_model_path, pose_detector_config_path);
    this->detection_result = new gusto_detector2d::DetectionResult();
    this->pose_result = new gusto_humanpose::KeyPoint2DResult();
    // this->detection_result = std::make_unique<gusto_detector2d::DetectionResult>();
    // this->pose_result = std::make_unique<gusto_humanpose::KeyPoint2DResult>();

    this->num_frames = 0;
    this->detect_interval = detect_interval;
}
HumanPoseExtractor2D::~HumanPoseExtractor2D() {}

GUSTO_RET HumanPoseExtractor2D::DetectPose(const cv::Mat& image)
{
    cv::Mat frame_resize, pose_input_copy;
    this->scale = LetterBoxImage(image, frame_resize, cv::Size(320, 320), 32, cv::Scalar(128,128,128), true);

    pose_input_copy = frame_resize.clone();

    if (this->num_frames == 0 || this->num_frames % this->detect_interval == 0){
        this->num_frames = 0;

        cv::Mat detector_input_copy = frame_resize.clone();
        // this->detection_result->boxes.clear();
        auto output = human_detector->forward(detector_input_copy);
        this->detection_result->boxes = static_cast<gusto_detector2d::DetectionResult*>(output.get())->boxes;
    }
    // auto output = human_detector->forward(detector_input_copy);

    // gusto_detector2d::DetectionResult* result = static_cast<gusto_detector2d::DetectionResult*>(output.get());
    auto dets_out = this->detection_result->boxes;
    for(size_t i = 0; i < dets_out.size(); i++){
        
        // RTMPose
        auto tmp_res = pose_detector->CropImageByDetectBox(pose_input_copy, dets_out[i]);
        auto pose_out = pose_detector->forward(tmp_res.first);
        gusto_humanpose::KeyPoint2DResult* pose_result = static_cast<gusto_humanpose::KeyPoint2DResult*>(pose_out.get());
        this->pose_result->keypoints = pose_result->keypoints;
        // gusto_humanpose::KeyPoint2DResult* pose_result = static_cast<gusto_humanpose::KeyPoint2DResult*>(pose_detector->forward(tmp_res.first).get());
        // for(size_t j = 0; j < keypoints.size(); j++){
            // pose_detector->draw_single_person_keypoints(image, keypoints, scale);
        // }

        break; //debug for one person
    }
    this->num_frames++;
    return GustoStatus::ERR_OK;
}

GUSTO_RET HumanPoseExtractor2D::Display(cv::Mat& image, bool display_box, bool display_keypoints)
{
    auto dets_out = this->detection_result->boxes;
    for(size_t i = 0; i < dets_out.size(); i++){
        if(display_box){
            cv::rectangle(image, cv::Point(dets_out[i].x1 * scale, dets_out[i].y1 * scale), cv::Point(dets_out[i].x2 * scale, dets_out[i].y2 * scale), cv::Scalar(0, 255, 0), 2);
        }
        
        if (display_keypoints){
            for(size_t j = 0; j < this->pose_result->keypoints.size(); j++){
                pose_detector->draw_single_person_keypoints(image, this->pose_result->keypoints, this->scale);
            }
        }


        break; //debug for one person
    }
    return GustoStatus::ERR_OK;
}