#include "csrt3d.h"
#include <filesystem>
#include <chrono>

GustoModelTarget::GustoModelTarget(const int height, const int width): height(height), width(width) {}


const Eigen::Matrix4f TRANSFORM_OPENCV_OPENGL = 
    (Eigen::Matrix4f() << 
        1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1 ).finished();

// cv::Mat convert_numpy_cvmat(const py::array_t<uint8_t>& input, bool gray_scale) {
//     if(!gray_scale && input.ndim() != 3) 
//         throw std::runtime_error("Input should be 3D numpy array");
//     if(gray_scale && input.ndim() != 2) 
//         throw std::runtime_error("Input should be 2D numpy array");
//     // get buffer info
//     py::buffer_info buf = input.request();
//     cv::Mat image(buf.shape[0], buf.shape[1], (gray_scale ? CV_8UC1 : CV_8UC3), (uint8_t*)buf.ptr);
//     return image;
// }


cv::Point2i calculate_pose_uv(const Eigen::Matrix4f& pose, const Eigen::Matrix3f& K) {
    Eigen::Vector3f p = pose.block<3, 1>(0, 3);
    Eigen::Vector3f puv = K * p;
    // std::cout << "p: " << p.transpose() << std::endl;
    // std::cout << "puv: " << puv.transpose() << std::endl;
    return cv::Point2i(puv[0] / puv[2], puv[1] / puv[2]);
}


RegionModel::RegionModel(const std::string& name,
    const std::string& geometry_path, 
    const std::string& meta_path_,
    const float unit_in_meter, 
    const float shpere_radius, 
    const bool from_opengl,
    const Eigen::Matrix4f& geometry2body_,
    const float threshold_on_init,
    const float threshold_on_track,
    const float kl_threshold,
    const bool debug_visualize
) : name_(name), threshold_on_init_(threshold_on_init), threshold_on_track_(threshold_on_track), kl_threshold_(kl_threshold),
    debug_visualize_(debug_visualize)
{
    Eigen::Matrix4f geometry2body = geometry2body_;
    if (from_opengl) {
        geometry2body = TRANSFORM_OPENCV_OPENGL * geometry2body_;
    }
    // make transform
    srt3d::Transform3fA geometry2body_pose(geometry2body);
    // make body geometry
    body_ptr_ = std::make_shared<srt3d::Body>(name, 
        geometry_path, unit_in_meter, true, true, geometry2body_pose);

    // set model
    auto meta_path = meta_path_;
    if (meta_path.empty()) {
        meta_path = geometry_path + ".meta";
    }
    model_ptr_ = std::make_shared<srt3d::Model>(name, body_ptr_, meta_path, shpere_radius);
}

void RegionModel::Setup() {
    model_ptr_->SetUp();
}

void RegionModel::set_kl_threshold(const float kl_threshold) {
    kl_threshold_ = kl_threshold;
    region_modality_ptr_->set_kl_threshold(kl_threshold);
}

void RegionModel::reset_pose(const Eigen::Matrix4f& pose) {
    srt3d::Transform3fA body2world_pose(pose);
    body_ptr_->set_body2world_pose(body2world_pose);
    last_pose_ = pose;
}


const Eigen::Matrix4f RegionModel::pose() {
    return body_ptr_->body2world_pose().matrix();
}


const Eigen::Matrix4f RegionModel::pose_gl() {
    Eigen::Matrix4f pose_cv = body_ptr_->body2world_pose().matrix();
    Eigen::Matrix4f pose_gl = TRANSFORM_OPENCV_OPENGL * pose_cv;
    return pose_gl;
}


bool RegionModel::validate() {
    // while tracking
    if (state_ == 0) { // stage: init
        if (conf_ >= threshold_on_track_) last_pose_ = pose();
        else {
            reset_pose(last_pose_);
            state_ = 1;
        }
    } else if (state_ == 1) { // stage: on tracking
        if (conf_ >= threshold_on_init_) {
            last_pose_ = pose();
            state_ = 0;
        } else reset_pose(last_pose_); 
    }

    return state_ == 0;
}

float RegionModel::valid_percentage() {
    int len_data_lines = region_modality_ptr_->len_data_lines();
    return (float)len_data_lines / (float)region_modality_ptr_->get_n_lines();
}



RegionTracker::RegionTracker(
    const int imwidth, const int imheight, 
    const Eigen::Matrix3f& K,
    const float fx_, const float fy_,
    const float cx_, const float cy_,
    const int corr_update_iter, const int pose_update_iter)
{
    // make tracker
    tracker_ptr_ = std::make_shared<srt3d::Tracker>("tracker");
    tracker_ptr_->set_n_corr_iterations(corr_update_iter);
    tracker_ptr_->set_n_update_iterations(pose_update_iter);

    /// set camera =======================================================
    // make virtual camera
    float fx = fx_, fy = fy_, cx = cx_, cy = cy_;
    camera_ptr_ = std::make_shared<srt3d::VirtualCamera>("vcam");

    if (!K.isZero()) {
        fx = K(0, 0);
        fy = K(1, 1);
        cx = K(0, 2);
        cy = K(1, 2);
    } else {
        if (cx * cy == 0.0) {
            cx = (float)imwidth / 2.0f;
            cy = (float)imheight / 2.0f;
        }
        if (fx == 0.0) {
            fx = std::max((float)imwidth, (float)imheight);
            fy = fx;
        }
        if (fy == 0.0) {
            fy = fx;
        }
    }

    std::cout << "Camera intrinsics: " << fx << " " << fy << " " << cx << " " << cy << std::endl;

    srt3d::Intrinsics intrinsics {
        fx, fy, cx, cy, imwidth, imheight
    };
    camera_ptr_->set_intrinsics(intrinsics);
    // set K
    K_ << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
}

void RegionTracker::add_model(const std::shared_ptr<RegionModel>& region_model) {
    // set region model
    auto region_modality_ptr = std::make_shared<srt3d::RegionModality>(
        region_model->name_, region_model->body_ptr_, region_model->model_ptr_, camera_ptr_);
    tracker_ptr_->AddRegionModality(region_modality_ptr);
    region_model->region_modality_ptr_ = region_modality_ptr;
    region_model->region_modality_ptr_->set_kl_threshold(region_model->kl_threshold_);
    region_model->region_modality_ptr_->debug_visualize_ = region_model->debug_visualize_;
    models_.push_back(std::move(region_model));
}


void RegionTracker::setup() {
    if (!tracker_ptr_->set_up()) {
        tracker_ptr_->SetUpTracker();
    }
}

// void RegionTracker::track(const py::array_t<uint8_t>& image, std::optional<py::array_t<uint8_t>> mask) 
void RegionTracker::track(const cv::Mat& image, std::optional<cv::Mat> mask)
{
    if (!tracker_ptr_->set_up()) {
        tracker_ptr_->SetUpTracker();
    }
    // get rgb image
    // cv::Mat cv_image = convert_numpy_cvmat(image, false);
    // cv::Mat cv_image = image.clone();
    cv::Mat cv_image_bgr;
    cv::cvtColor(image, cv_image_bgr, cv::COLOR_RGB2BGR);
    // get mask
    cv::Mat occlusion_mask;
    if (mask.has_value()) {
        // occlusion_mask = convert_numpy_cvmat(mask.value() , true);
        occlusion_mask = mask.value().clone();
    }
    // push image
    camera_ptr_->PushImage(cv_image_bgr);
    // main tracking loop
    tracker_ptr_->TrackIter(occlusion_mask);
    auto st = std::chrono::steady_clock::now();

    // evaluate distribution
    std::vector<float> conf_list = tracker_ptr_->EvaluateDistribution();
    for(int i = 0; i < conf_list.size(); i++) {
        models_[i]->conf_ = conf_list[i];
        models_[i]->validate();
        models_[i]->uv_ = calculate_pose_uv(models_[i]->pose(), K_);
    }

    double ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - st).count();
    std::cout << "Evaluation time: " << ms << "ms" << std::endl;
}

void RegionTracker::terminate() {
    for(auto &model : models_) {
        model.reset();
    }
    models_.clear();
    tracker_ptr_.reset();
    camera_ptr_.reset();
}


#if !defined(__DISABLE_OPENGL__)
RegionRenderer::RegionRenderer(const std::shared_ptr<RegionTracker>& region_tracker) {
    region_tracker_ = region_tracker;
    camera_ptr_ = region_tracker_->camera_ptr_;
    // setup viewer
    renderer_geometry_ptr_ = 
        std::make_shared<srt3d::RendererGeometry>("renderer geometry");
    // set body pointer
    for(auto &model_ptr : region_tracker_->models_) {
        renderer_geometry_ptr_->AddBody(model_ptr->body_ptr_);
    }
    viewer_ptr_ =
        std::make_shared<srt3d::NormalViewer>("normal viewer", camera_ptr_ ,renderer_geometry_ptr_);
    camera_ptr_->SetUp();
    renderer_geometry_ptr_->SetUp();
    viewer_ptr_->SetUp();
}

// py::array_t<uint8_t> RegionRenderer::render() {
std::optional<cv::Mat> RegionRenderer::render() {
    std::error_code errorCode;
    if(!camera_ptr_->UpdateImage()) return std::nullopt;
    if(!viewer_ptr_->UpdateViewer(0)) return std::nullopt;
    // render
    cv::Mat normal_image = viewer_ptr_->viewer_image();
    // convert cv::Mat to numpy array
    // py::array_t<uint8_t> normal_image_np = py::array_t<uint8_t>(
    //     {normal_image.rows, normal_image.cols, normal_image.channels()},
    //     {normal_image.cols * normal_image.channels(), normal_image.channels(), 1},
    //     normal_image.data
    // );
    return normal_image;
}
#endif // __DISABLE_OPENGL__