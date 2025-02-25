#include "multi_stage_face_geometry_3d.h"


// cv::Mat draw_axis(cv::Mat img, cv::Mat R, cv::Mat t, cv::Mat K, cv::Point2f origin) {
//     // unit is mm
//     cv::Mat rotV;
//     cv::Rodrigues(R, rotV);
//     std::vector<cv::Point3f> points = { {10, 0, 0}, {0, 10, 0}, {0, 0, 10}, {0, 0, 0} };
//     std::vector<cv::Point2f> axisPoints;
//     cv::projectPoints(points, rotV, t, K, cv::Mat(), axisPoints);

//     for (auto& point : axisPoints) {
//         point.x = img.cols - point.x;
//     }

//     // Compute direction vectors
//     cv::Point2f x_axis = axisPoints[0] - axisPoints[3];
//     cv::Point2f y_axis = axisPoints[1] - axisPoints[3];
//     cv::Point2f z_axis = axisPoints[2] - axisPoints[3];

//     // draw
//     // cv::line(img, axisPoints[3], axisPoints[0], cv::Scalar(255, 0, 0), 3); // X-axis in red
//     // cv::line(img, axisPoints[3], axisPoints[1], cv::Scalar(0, 255, 0), 3); // Y-axis in green
//     // cv::line(img, axisPoints[3], axisPoints[2], cv::Scalar(0, 0, 255), 3); // Z-axis in blue

//     // Use origin to draw the axes
//     // std::cout << "Origin: " << origin << std::endl;
//     cv::circle(img, origin, 5, cv::Scalar(255, 255, 255), -1);
//     cv::line(img, origin, origin + x_axis, cv::Scalar(255, 0, 0), 3); // X-axis in red
//     cv::line(img, origin, origin + y_axis, cv::Scalar(0, 255, 0), 3); // Y-axis in green
//     cv::line(img, origin, origin + z_axis, cv::Scalar(0, 0, 255), 3); // Z-axis in blue

//     return img;
// }




FaceGeometryTracker3D::FaceGeometryTracker3D(
    const std::string& face_detector_path,
    const std::string& face_detector_config_path,
    const std::string& face_landmarker_path,
    const std::string& face_landmarker_config_path,
    const std::string face_GeometryPipelineMetadata,
    int detect_interval) : detect_interval(detect_interval), num_frames(0) {

    face_detector = std::make_unique<gusto_mp_face::FaceDetector>(face_detector_path, face_detector_config_path);
    face_landmarker = std::make_unique<gusto_mp_face::FaceLandmarker>(face_landmarker_path, face_landmarker_config_path);
    face_mesh_calculator = std::make_unique<gusto_face_geometry::FaceMeshCalculator>();
    face_mesh_calculator->Open(face_GeometryPipelineMetadata);
    
}
FaceGeometryTracker3D::~FaceGeometryTracker3D() {}

GUSTO_RET FaceGeometryTracker3D::Detect(const cv::Mat& frame, bool display_keypoints, bool display_coord){
    
    // cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    // auto [boxes, scores, indices, indices_cls] = face_detector.forward(frame);
    // this->rendered_frame = frame.clone();
    auto ret = face_detector->forward(frame);
    auto* face_detector_result = dynamic_cast<gusto_mp_face::MediaPipeDetectorResult*>(ret.get());
    // std::vector<gusto_face_geometry::NormalizedLandmarkList> multi_face_landmarks;
    for (size_t i = 0; i < face_detector_result->boxes.size(); i++){
    // for(size_t idx = 0; idx < indices.size(); idx++) {
        // std::vector<int> box_to_crop = {
        //     static_cast<int>(boxes[indices[idx]].y1 * frame.size[0]),
        //     static_cast<int>(boxes[indices[idx]].x1 * frame.size[1]),
        //     static_cast<int>(boxes[indices[idx]].y2 * frame.size[0]),
        //     static_cast<int>(boxes[indices[idx]].x2 * frame.size[1]), 
        // }; 
        std::vector<int> box_to_crop = {
            static_cast<int>(face_detector_result->boxes[i].y1 * frame.size[0]),
            static_cast<int>(face_detector_result->boxes[i].x1 * frame.size[1]),
            static_cast<int>(face_detector_result->boxes[i].y2 * frame.size[0]),
            static_cast<int>(face_detector_result->boxes[i].x2 * frame.size[1]), 
        };
        // cv::Mat cropped_face = face_landmarker.crop_face(frame, box_to_crop);
        auto [cropped_face, box_to_crop_with_margin] = face_landmarker->crop_face(frame, box_to_crop);
        auto ret = face_landmarker->forward(cropped_face);
        auto* face_landmarker_result = dynamic_cast<gusto_mp_face::MediapipeFaceLandmarkResult*>(ret.get());
        auto points = face_landmarker_result->points;
        auto score = face_landmarker_result->score;

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
        if (display_keypoints){
            this->rendered_frame = face_landmarker->draw_points(frame, points, cv::Point(box_to_crop_with_margin[1], box_to_crop_with_margin[0]));
        }
    }

    auto [multi_pose_mat, process_status] = face_mesh_calculator->Process(std::make_pair(frame.size[0], frame.size[1]), multi_face_landmarks);
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);

    const float fov = 70.0;
    const float width = frame.size[1];
    const float height = frame.size[0];
    // const float fx = (width / 2) / np.tan(np.deg2rad(fov)/2)
    const float fx = (width / 2) / std::tan(3.1415 * fov / 360.0);
    // const float fx = 500.0;
    const float fy = fx;
    // const float cx = height / 2;
    // const float cy = width / 2;
    const float cx = width / 2;
    const float cy = height / 2;
    
    K.at<float>(0, 0) = fx; // Focal length in x direction
    K.at<float>(1, 1) = fy; // Focal length in y direction
    K.at<float>(0, 2) = cx; // Principal point x-coordinate
    K.at<float>(1, 2) = cy; // Principal point y-coordinate
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
    return GustoStatus::ERR_OK;
}

cv::Mat FaceGeometryTracker3D::GetRenderedFrame(){
    return this->rendered_frame;
}

