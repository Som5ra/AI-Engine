#include "csrt3d.h"
#include <filesystem>
#include <chrono>
#include <opencv2/opencv.hpp>


/*
f_approx = (width/2)/np.tan(np.deg2rad(args.fov)/2)
intrinsics_approx = {
    'fu': f_approx,
	'fv': f_approx,
	'ppu': width/2,
	'ppv': height/2,
	'width': width,
	'height': height,
}

cam_intrinsics = {
	'intrinsics_color': intrinsics_approx,
	'quat_d_c_xyzw': [0,0,0,1],
    'trans_d_c': [0,0,0],
}

*/
int main()
{
    // setup camera
    // const int height = 480;
    const int height = 448;
    // const int width = 640;
    const int width = 800;
    const float fov = 50.0;

    // const float fx = (width / 2) / np.tan(np.deg2rad(fov)/2)
    const float fx = (width / 2) / std::tan(3.1415 * fov / 360.0);
    // const float fx = 500.0;
    const float fy = fx;
    // const float cx = height / 2;
    // const float cy = width / 2;
    const float cx = width / 2;
    const float cy = height / 2;
    const Eigen::Matrix3f K = (Eigen::Matrix3f() << fx, 0, cx, 0, fy, cy, 0, 0, 1).finished();
    // const Eigen::Matrix3f K = Eigen::Matrix3f::Zero();

    //setup CAD model
    const std::string model_name = "Bruni-woband";
    const std::string model_path = "/media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/StreamingAssets/Bruni-woband/Bruni-woband.obj";
    const std::string meta_path = "/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/123.meta";

    const float unit_in_meter = 2.0;
    const float sphere_radius = 0.8;
    const bool from_opengl = false;
    const Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    const float threshold_on_init = 0.8;
    const float threshold_on_track = 0.8;
    const float kl_threshold = 1.0;
    const bool debug_visualize = false;

    const Eigen::Matrix4f init_pose = (Eigen::Matrix4f() << 
        -1, 0, 0, 0,
        0, 0, -1, 0,
        0, -1, 0, 0.8,
        0, 0, 0, 1
    ).finished();


    std::shared_ptr<RegionModel> model = std::make_shared<RegionModel>(
        model_name,
        model_path,
        meta_path,
        unit_in_meter,
        sphere_radius,
        from_opengl,
        transform,
        threshold_on_init,
        threshold_on_track,
        kl_threshold,
        debug_visualize
    );

    // setup tracker
    const int corr_iter = 7;
    const int pose_iter = 2;

    std::shared_ptr<RegionTracker> tracker = std::make_shared<RegionTracker>(width, height, K, fx, fy, cx, cy, corr_iter, pose_iter);
    tracker->add_model(model);
    tracker->setup();

    // setup renderer
    // RegionRenderer renderer = RegionRenderer(std::make_shared<RegionTracker>(tracker));
    std::shared_ptr<RegionRenderer> renderer = std::make_shared<RegionRenderer>(tracker);
    // set initial pose
    model->reset_pose(init_pose);
    


    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FPS, 30);
    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        
        cv::Mat image_rgb;
        cv::cvtColor(frame, image_rgb, cv::COLOR_BGR2RGB);
        // cv::resize(image_rgb, image_rgb, cv::Size(width, height));
        std::cout << "image_rgb: " << image_rgb.size() << std::endl;
        auto st = std::chrono::steady_clock::now();

        tracker->track(image_rgb, std::nullopt);
        auto et = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count();
        std::cout << "elapsed_time: " << elapsed_time << "ms" << std::endl;
        std::cout << model->pose() << std::endl;

        float conf = model->conf_;
        cv::Point2i pose_uv = model->uv_;
        std::cout << "confidence: " << conf << " pose_uv: " << pose_uv << std::endl;

        auto res = renderer->render();
        auto rendered_image = std::move(*res);
        cv::putText(rendered_image, "confidence: " + std::to_string(conf), pose_uv, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        cv::imshow("rendered_image", rendered_image);
        cv::waitKey(1);
    }
    

    return 0;
}


// #include "csrt3d.h"
// #include <filesystem>
// #include <chrono>
// #include <opencv2/opencv.hpp>


// /*
// f_approx = (width/2)/np.tan(np.deg2rad(args.fov)/2)
// intrinsics_approx = {
//     'fu': f_approx,
// 	'fv': f_approx,
// 	'ppu': width/2,
// 	'ppv': height/2,
// 	'width': width,
// 	'height': height,
// }

// cam_intrinsics = {
// 	'intrinsics_color': intrinsics_approx,
// 	'quat_d_c_xyzw': [0,0,0,1],
//     'trans_d_c': [0,0,0],
// }

// */
// int main()
// {
//     // setup camera
//     const int height = 480;
//     const int width = 640;
//     const float fov = 60.0;

//     // const float fx = (width / 2) / np.tan(np.deg2rad(fov)/2)
//     const float fx = (width / 2) / std::tan(3.1415 * fov / 360.0);
//     // const float fx = 500.0;
//     const float fy = fx;
//     const float cx =  width / 2;
//     const float cy = height / 2;
//     const Eigen::Matrix3f K = (Eigen::Matrix3f() << fx, 0, cx, 0, fy, cy, 0, 0, 1).finished();
//     // const Eigen::Matrix3f K = Eigen::Matrix3f::Zero();

//     //setup CAD model
//     const std::string model_name = "Bruni-woband";
//     const std::string model_path = model_name + "/" + model_name + ".obj";
//     const std::string meta_path = model_name + "/" + model_name + ".meta";

//     const float unit_in_meter = 2.0;
//     const float sphere_radius = 0.8;
//     const bool from_opengl = false;
//     const Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
//     const float threshold_on_init = 0.8;
//     const float threshold_on_track = 0.8;
//     const float kl_threshold = 1.0;
//     const bool debug_visualize = false;

//     const Eigen::Matrix4f init_pose = (Eigen::Matrix4f() << 
//         -1, 0, 0, 0,
//         // -1, 0, 0, 0.022408963585434174,
//         0, 0, -1, 0,
//         // 0, 0, -1, 0.03373155305692199,
//         0, -1, 0, 0.8,
//         0, 0, 0, 1
//     ).finished();


//     std::shared_ptr<RegionModel> model = std::make_shared<RegionModel>(
//         model_name,
//         model_path,
//         meta_path,
//         unit_in_meter,
//         sphere_radius,
//         from_opengl,
//         transform,
//         threshold_on_init,
//         threshold_on_track,
//         kl_threshold,
//         debug_visualize
//     );

//     // setup tracker
//     const int corr_iter = 7;
//     const int pose_iter = 2;

//     std::shared_ptr<RegionTracker> tracker = std::make_shared<RegionTracker>(width, height, K, fx, fy, cx, cy, corr_iter, pose_iter);
//     tracker->add_model(model);
//     tracker->setup();

//     // setup renderer
//     // RegionRenderer renderer = RegionRenderer(std::make_shared<RegionTracker>(tracker));
//     std::shared_ptr<RegionRenderer> renderer = std::make_shared<RegionRenderer>(tracker);
//     // set initial pose
//     model->reset_pose(init_pose);
    


//     cv::VideoCapture cap(0);
//     cap.set(cv::CAP_PROP_FPS, 30);
//     while (true)
//     {
//         cv::Mat frame;
//         cap >> frame;
//         if (frame.empty())
//         {
//             break;
//         }
//         cv::resize(frame, frame, cv::Size(width, height));
//         cv::Mat image_rgb;
//         cv::cvtColor(frame, image_rgb, cv::COLOR_BGR2RGB);
//         std::cout << "image shape: " << image_rgb.size() << std::endl;
//         tracker->track(image_rgb, std::nullopt);
//         float conf = model->conf_;
//         cv::Point2i pose_uv = model->uv_;
//         std::cout << "confidence: " << conf << " pose_uv: " << pose_uv << std::endl;

//         auto res = renderer->render();
//         auto rendered_image = std::move(*res);
//         cv::putText(rendered_image, "confidence: " + std::to_string(conf), pose_uv, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

//         cv::imshow("rendered_image", rendered_image);
//         cv::waitKey(1);
//     }
    

//     return 0;
// }