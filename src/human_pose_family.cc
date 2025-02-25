#include "human_pose_family.h"

namespace gusto_humanpose{

// std::map<std::string, model_lib> MODEL_NAME_LIB_MAPPER = {
//     {"rtmo-s", model_lib::RTMO_S}
// };

// std::unique_ptr<basic_model_config> fetch_model_config(std::unique_ptr<basic_model_config>& base_config){
//     // std::unique_ptr<base_model_config> _config(new base_model_config());
//     std::cout <<  "input model name: " << base_config->model_name << std::endl;
//     model_lib model_type;
//     try{
//         model_type = MODEL_NAME_LIB_MAPPER.at(base_config->model_name);
//         // _config->model_name = _model_name;
//         // _config->model_type = MODEL_NAME_LIB_MAPPER[_model_name];
//     }catch(const std::exception& e){
//         std::cerr << "input model name is not valid! " << std::endl;
//         return nullptr;
//         // return _config;
//     }

//     if(model_type == model_lib::RTMO_S){
//         base_config->model_path = "rtmo-s.onnx";
//         base_config->input_size = std::make_pair(640, 640);
//         base_config->class_mapper = {
//             // to be written
//         };
//     }

//     std::cout << "model path: " << base_config->model_path << std::endl;
//     return std::move(base_config);
// }

RTMPose::RTMPose(const std::string& model_path, const std::string& config_path)
    : BaseONNX(model_path, config_path) {
}
RTMPose::RTMPose(std::unique_ptr<basic_model_config> _config)
    : BaseONNX(std::move(_config)) {
}

cv::Mat RTMPose::GetAffineTransform(float center_x, float center_y, float scale_width, float scale_height, int output_image_width, int output_image_height, bool inverse)
{
	// solve the affine transformation matrix
	/* 求解仿射变换矩阵 */

	// get the three points corresponding to the source picture and the target picture
	// 获取源图片与目标图片的对应的三个点
	cv::Point2f src_point_1;
	src_point_1.x = center_x;
	src_point_1.y = center_y;

	cv::Point2f src_point_2;
	src_point_2.x = center_x;
	src_point_2.y = center_y - scale_width * 0.5;

	cv::Point2f src_point_3;
	src_point_3.x = src_point_2.x - (src_point_1.y - src_point_2.y);
	src_point_3.y = src_point_2.y + (src_point_1.x - src_point_2.x);


	float alphapose_image_center_x = output_image_width / 2;
	float alphapose_image_center_y = output_image_height / 2;

	cv::Point2f dst_point_1;
	dst_point_1.x = alphapose_image_center_x;
	dst_point_1.y = alphapose_image_center_y;

	cv::Point2f dst_point_2;
	dst_point_2.x = alphapose_image_center_x;
	dst_point_2.y = alphapose_image_center_y - output_image_width * 0.5;

	cv::Point2f dst_point_3;
	dst_point_3.x = dst_point_2.x - (dst_point_1.y - dst_point_2.y);
	dst_point_3.y = dst_point_2.y + (dst_point_1.x - dst_point_2.x);


	cv::Point2f srcPoints[3];
	srcPoints[0] = src_point_1;
	srcPoints[1] = src_point_2;
	srcPoints[2] = src_point_3;

	cv::Point2f dstPoints[3];
	dstPoints[0] = dst_point_1;
	dstPoints[1] = dst_point_2;
	dstPoints[2] = dst_point_3;

	// get affine matrix
	// 获取仿射矩阵
	cv::Mat affineTransform;
	if (inverse)
	{
		affineTransform = cv::getAffineTransform(dstPoints, srcPoints);
	}
	else
	{
		affineTransform = cv::getAffineTransform(srcPoints, dstPoints);
	}

	return affineTransform;
}

std::pair<cv::Mat, cv::Mat> RTMPose::CropImageByDetectBox(const cv::Mat& input_image, const GustoRect& box)
{
    float left = box.x1;
    float top = box.y1;
    float right = box.x2;
    float bottom = box.y2;
	std::pair<cv::Mat, cv::Mat> result_pair;

	if (!input_image.data)
	{
		return result_pair;
	}

	// if (!box.IsValid())
    // if (!(left != -1 && top != -1 && right != -1 && bottom != -1 && score != -1 && label != -1))
	if (!(left != -1 && top != -1 && right != -1 && bottom != -1))
	{
		return result_pair;
	}

	// deep copy
	// 深拷贝
	cv::Mat input_mat_copy;
	input_image.copyTo(input_mat_copy);

	// calculate the width, height and center points of the human detection box
	// 计算人体检测框的宽、高以及中心点
	int box_width = right - left;
	int box_height = bottom - top;
	int box_center_x = left + box_width / 2;
	int box_center_y = top + box_height / 2;

	float aspect_ratio = 192.0 / 256.0;

	// adjust the width and height ratio of the size of the picture in the RTMPOSE input
	// 根据rtmpose输入图片大小的宽高比例进行调整
	if (box_width > (aspect_ratio * box_height))
	{
		box_height = box_width / aspect_ratio;
	}
	else if (box_width < (aspect_ratio * box_height))
	{
		box_width = box_height * aspect_ratio;
	}

	float scale_image_width = box_width * 1.25;
	float scale_image_height = box_height * 1.25;

	// get the affine matrix
	// 获取仿射矩阵
	cv::Mat affine_transform = RTMPose::GetAffineTransform(
		box_center_x,
		box_center_y,
		scale_image_width,
		scale_image_height,
		192,
		256
	);
	
	cv::Mat affine_transform_reverse = RTMPose::GetAffineTransform(
		box_center_x,
		box_center_y,
		scale_image_width,
		scale_image_height,
		192,
		256,
		true
	);

	// affine transform
	// 进行仿射变换
	cv::Mat affine_image;
	cv::warpAffine(input_mat_copy, affine_image, affine_transform, cv::Size(192, 256), cv::INTER_LINEAR);
	//cv::imwrite("affine_img.jpg", affine_image);
    this->affine_transform_reverse = affine_transform_reverse;
	result_pair = std::make_pair(affine_image, affine_transform_reverse);
	return result_pair;
}



// std::vector<float> RTMPose::preprocess_img(const cv::Mat& image) {
//     cv::Mat frame;
//     frame = image.clone();
//     cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
//     float ratio = std::min(static_cast<float>(_config->input_size.first) / image.rows, static_cast<float>(_config->input_size.second) / image.cols);
//     this->preprocess_ratio = ratio;
//     cv::Mat resized_img;
//     cv::resize(frame, resized_img, cv::Size(static_cast<int>(image.cols * ratio), static_cast<int>(image.rows * ratio)), 0, 0, cv::INTER_LINEAR);

//     cv::Rect roi(cv::Point(0, 0), resized_img.size());
//     cv::Mat padded_img(_config->input_size.first, _config->input_size.second, CV_8UC3, cv::Scalar(114, 114, 114));
//     resized_img.copyTo(padded_img(roi));
//     // cv::resize(frame, frame, cv::Size(_config->input_size.second, _config->input_size.first), 0, 0, cv::INTER_LINEAR);
//     // frame.convertTo(frame, CV_32FC3, 1.0 / 127.5, -1); 
//     // padded_img.convertTo(frame, CV_32FC3, 1.0, 0.0); 
//     std::vector<cv::Mat> rgbsplit;
//     cv::split(padded_img, rgbsplit);

//     std::vector<float> input_tensor_values(inputTensorSize);


//     int h = rgbsplit[0].size[0];
//     int w = rgbsplit[0].size[1];
//     #if !defined(BUILD_PLATFORM_WINDOWS) && !defined(BUILD_PLATFORM_IOS)
//     #pragma omp parallel for num_threads(2)
//     #pragma omp parallel for
//     #endif
//     for (int i = 0; i < h; i++) {
//         for (int j = 0; j < w; j++) {
//             for (int k = 0; k < rgbsplit.size(); k++) {
//                 // std::cout << "i: " <<  i << " j: " << j << " k: " << k << " value: " << rgbsplit[k].at<float>(i, j)<< std::endl;
//                 // std::cout << "i: " <<  i << " j: " << j << " k: " << k << " value: " << static_cast<float>(rgbsplit[k].at<uint8_t>(i, j)) << std::endl;
//                 input_tensor_values[k * h * w + i * w + j] = static_cast<float>(rgbsplit[k].at<uint8_t>(i, j)); // CHW
//             }
//         }
//     }

//     return input_tensor_values;
// }


std::unique_ptr<PostProcessResult> RTMPose::forward(const cv::Mat& image){
    // std::vector<float> input_tensor_values = preprocess_img(image);
    // std::pair<cv::Mat, cv::Mat> crop_result_pair = CropImageByDetectBox(image, box, score, label);
	// cv::Mat crop_mat = crop_result_pair.first;
	// cv::Mat affine_transform_reverse = crop_result_pair.second;
	// cv::Mat crop_mat_copy;
	// crop_mat.copyTo(crop_mat_copy);

    auto input_tensor_values = preprocess(image);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), inputTensorSize, input_shape[0].data(), input_shape[0].size());
    Ort::RunOptions run_options{};
    std::vector<Ort::Value> output_tensors = ort_session.Run(run_options, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_names.size());
    // std::vector<Ort::Value> output_tensors;
    std::unique_ptr<KeyPoint2DResult> result = std::make_unique<KeyPoint2DResult>();
    result->keypoints = postprocess(output_tensors, 0.5);
    // return output_tensors;
    return result;
}

std::vector<std::tuple<int, int, int>> RTMPose::postprocess(const std::vector<Ort::Value>& output_tensors, float threshold){ 

    std::vector<int64_t> simcc_x_dims = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	std::vector<int64_t> simcc_y_dims = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

	assert(simcc_x_dims.size() == 3 && simcc_y_dims.size() == 3);

	int batch_size = simcc_x_dims[0] == simcc_y_dims[0] ? simcc_x_dims[0] : 0;
	int joint_num = simcc_x_dims[1] == simcc_y_dims[1] ? simcc_x_dims[1] : 0;
	int extend_width = simcc_x_dims[2];
	int extend_height = simcc_y_dims[2];
	// std::cout << "batch_size: " << batch_size << std::endl;
	// std::cout << "joint_num: " << joint_num << std::endl;
	// std::cout << "extend_width: " << extend_width << std::endl;
	// std::cout << "extend_height: " << extend_height << std::endl;
	const float* simcc_x_result = output_tensors[0].GetTensorData<float>();
	const float* simcc_y_result = output_tensors[1].GetTensorData<float>();

	std::vector<std::tuple<int, int, int>> keypoints;

	for (int i = 0; i < joint_num; ++i)
	{
		// find the maximum and maximum indexes in the value of each Extend_width length
		// 在每一个extend_width长度的数值中找到最大值以及最大值的索引
		auto x_biggest_iter = std::max_element(simcc_x_result + i * extend_width, simcc_x_result + i * extend_width + extend_width);
		int max_x_pos = std::distance(simcc_x_result + i * extend_width, x_biggest_iter);
		int pose_x = max_x_pos / 2;
		float score_x = *x_biggest_iter;

		// find the maximum and maximum indexes in the value of each exten_height length
		// 在每一个extend_height长度的数值中找到最大值以及最大值的索引
		auto y_biggest_iter = std::max_element(simcc_y_result + i * extend_height, simcc_y_result + i * extend_height + extend_height);
		int max_y_pos = std::distance(simcc_y_result + i * extend_height, y_biggest_iter);
		int pose_y = max_y_pos / 2;
		float score_y = *y_biggest_iter;

		//float score = (score_x + score_y) / 2;
		float score = std::max(score_x, score_y);
		// PosePoint temp_point;
		// temp_point.x = int(pose_x);
		// temp_point.y = int(pose_y);
		// temp_point.score = score;
		// pose_result.emplace_back(temp_point);
        // keypoints.push_back({std::make_tuple(pose_x, pose_y, score > threshold ? 1 : 0)});

	// anti affine transformation to obtain the coordinates on the original picture
	// 反仿射变换获取在原始图片上的坐标
	// for (int i = 0; i < keypoints.size(); ++i)
	// {
        if (score > threshold){
			cv::Mat origin_point_Mat = cv::Mat::ones(3, 1, CV_64FC1);
			origin_point_Mat.at<double>(0, 0) = pose_x;
			origin_point_Mat.at<double>(1, 0) = pose_y;
			cv::Mat temp_result_mat = this->affine_transform_reverse * origin_point_Mat;
			// std::cout << "pose_x: " << temp_result_mat.at<double>(0, 0) << " pose_y: " << temp_result_mat.at<double>(1, 0) << " score: " << score << std::endl;

			keypoints.push_back(std::make_tuple(temp_result_mat.at<double>(0, 0), temp_result_mat.at<double>(1, 0), 1));
        }else{
            keypoints.push_back(std::make_tuple(-1, -1, 0));
        }
		
	}

	return keypoints;
    // // detector is kinda useless with rtmo-series models

    // const float* prob = output_tensors[1].GetTensorData<float>();
    // auto outputInfo = output_tensors[1].GetTensorTypeAndShapeInfo();
    // int batch_size = outputInfo.GetShape()[0];
    // int activated_person = outputInfo.GetShape()[1];
    // int kpts_num = outputInfo.GetShape()[2];
    // int kpts = outputInfo.GetShape()[3];
    // // std::cout << "batch_size: " << batch_size << std::endl;
    // // std::cout << "activated_person: " << activated_person << std::endl;
    // // std::cout << "width: " << kpts_num << std::endl;
    // // std::cout << "channels: " << kpts << std::endl;

    // // [Sombra] TODO: Support batch size > 1
    // assert(batch_size == 1);

    // std::vector<std::vector<std::tuple<int, int, int>>> keypoints;
    // // std::vector<float> single_person_keypoints_conf;
    // for(int i = 0; i < activated_person; i++){
    //     // std::cout << "person id: " << i << std::endl;
    //     std::vector<std::tuple<int, int, int>> single_person_keypoints;

    //     for(int j = 0; j < kpts_num; j++){

    //         int x = static_cast<int>(prob[0 + kpts * (j + kpts_num * i)] / this->preprocess_ratio);
    //         int y = static_cast<int>(prob[1 + kpts * (j + kpts_num * i)] / this->preprocess_ratio);
    //         float conf = prob[2 + kpts * (j + kpts_num * i)];
    //         // if (conf < threshold){
    //         //     continue;
    //         // }
    //         // std::cout << "x: " << x << " y: " << y << " conf: " << conf << std::endl;
    //         int exist = conf > threshold ? 1 : 0;
    //         single_person_keypoints.push_back(std::make_tuple(x, y, exist));
    //     }
    //     keypoints.push_back(single_person_keypoints);
    // }

    
    // return keypoints;
}


cv::Mat RTMPose::draw_single_person_keypoints(cv::Mat image, const std::vector<std::tuple<int, int, int>>& keypoints, float scale){
    for(size_t i = 0; i < keypoints.size(); i++){
        auto [x, y, exist] = keypoints[i];
        if (exist == 0){
            continue;
        }
        cv::circle(image, cv::Point(x * scale, y * scale), 5, coco17_mapper[i].second, 1);
        // std::cout << "keypoint: " << i << std::endl;
        // std::cout << coco17_mapper[i].first << std::endl;
        // cv::putText(image, coco17[i].first, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    for(size_t i = 0; i < coco17_skeleton.size(); i++){
        auto [start, end] = coco17_skeleton[i];
        auto [x1, y1, exist1] = keypoints[start];
        auto [x2, y2, exist2] = keypoints[end];
        if (exist1 == 0 || exist2 == 0){
            continue;
        }
        cv::line(image, cv::Point(x1 * scale, y1 * scale), cv::Point(x2 * scale, y2 * scale), coco17_mapper[i].second, 2);
    }
    return image;
}




} //namespace gusto_humanpose