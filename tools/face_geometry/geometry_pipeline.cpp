#include "utils.h"
#include "tools/face_geometry/geometry_pipeline.h"
#include "tools/face_geometry/face_geometry.h"
#include "tools/face_geometry/procrustes_solver.h"

namespace gusto_face_geometry {

class ScreenToMetricSpaceConverter {
 public:
  ScreenToMetricSpaceConverter(
      OriginPointLocation origin_point_location,      //
      InputSource input_source,                       //
      Eigen::Matrix3Xf&& canonical_metric_landmarks,  //
      Eigen::VectorXf&& landmark_weights,             //
      std::unique_ptr<ProcrustesSolver> procrustes_solver)
      : origin_point_location_(origin_point_location),
        input_source_(input_source),
        canonical_metric_landmarks_(std::move(canonical_metric_landmarks)),
        landmark_weights_(std::move(landmark_weights)),
        procrustes_solver_(std::move(procrustes_solver)) {}

  // Converts `screen_landmark_list` into `metric_landmark_list` and estimates
  // the `pose_transform_mat`.
  //
  // Here's the algorithm summary:
  //
  // (1) Project X- and Y- screen landmark coordinates at the Z near plane.
  //
  // (2) Estimate a canonical-to-runtime landmark set scale by running the
  //     Procrustes solver using the screen runtime landmarks.
  //
  //     On this iteration, screen landmarks are used instead of unprojected
  //     metric landmarks as it is not safe to unproject due to the relative
  //     nature of the input screen landmark Z coordinate.
  //
  // (3) Use the canonical-to-runtime scale from (2) to unproject the screen
  //     landmarks. The result is referenced as "intermediate landmarks" because
  //     they are the first estimation of the resuling metric landmarks, but are
  //     not quite there yet.
  //
  // (4) Estimate a canonical-to-runtime landmark set scale by running the
  //     Procrustes solver using the intermediate runtime landmarks.
  //
  // (5) Use the product of the scale factors from (2) and (4) to unproject
  //     the screen landmarks the second time. This is the second and the final
  //     estimation of the metric landmarks.
  //
  // (6) Multiply each of the metric landmarks by the inverse pose
  //     transformation matrix to align the runtime metric face landmarks with
  //     the canonical metric face landmarks.
  //
  // Note: the input screen landmarks are in the left-handed coordinate system,
  //       however any metric landmarks - including the canonical metric
  //       landmarks, the final runtime metric landmarks and any intermediate
  //       runtime metric landmarks - are in the right-handed coordinate system.
  //
  //       To keep the logic correct, the landmark set handedness is changed any
  //       time the screen-to-metric semantic barrier is passed.
  GUSTO_RET Convert(const NormalizedLandmarkList& screen_landmark_list,  //
                       const PerspectiveCameraFrustum& pcf,                 //
                       LandmarkList& metric_landmark_list,                  //
                       Eigen::Matrix4f& pose_transform_mat) const {
    if (screen_landmark_list.landmark_size() != canonical_metric_landmarks_.cols()) {
        std::cerr << "The number of landmarks doesn't match the number passed upon initialization!" << std::endl;
        return GustoStatus::ERR_GENERAL_INVALID_PARAMETER;
    }

    Eigen::Matrix3Xf screen_landmarks;
    ConvertLandmarkListToEigenMatrix(screen_landmark_list, screen_landmarks);

    ProjectXY(pcf, screen_landmarks);
    const float depth_offset = screen_landmarks.row(2).mean();

    // 1st iteration: don't unproject XY because it's unsafe to do so due to
    //                the relative nature of the Z coordinate. Instead, run the
    //                first estimation on the projected XY and use that scale to
    //                unproject for the 2nd iteration.
    Eigen::Matrix3Xf intermediate_landmarks(screen_landmarks);
    ChangeHandedness(intermediate_landmarks);


    float first_iteration_scale;
    if (EstimateScale(intermediate_landmarks, &first_iteration_scale) != GustoStatus::ERR_OK) {
        std::cerr << "Failed to estimate first iteration scale!" << std::endl;
        return GustoStatus::ERR_GENERAL_ERROR;
    }
    // 2nd iteration: unproject XY using the scale from the 1st iteration.
    intermediate_landmarks = screen_landmarks;
    MoveAndRescaleZ(pcf, depth_offset, first_iteration_scale,
                    intermediate_landmarks);
    UnprojectXY(pcf, intermediate_landmarks);
    ChangeHandedness(intermediate_landmarks);

    // For face detection input landmarks, re-write Z-coord from the canonical
    // landmarks.
    if (input_source_ == InputSource::FACE_DETECTION_PIPELINE) {
      Eigen::Matrix4f intermediate_pose_transform_mat;
      if (procrustes_solver_->SolveWeightedOrthogonalProblem(canonical_metric_landmarks_, intermediate_landmarks, landmark_weights_, intermediate_pose_transform_mat) != GustoStatus::ERR_OK) {
        std::cerr << "Failed to estimate pose transform matrix!" << std::endl;
        return GustoStatus::ERR_GENERAL_ERROR;
      }
      
      intermediate_landmarks.row(2) =
          (intermediate_pose_transform_mat *
           canonical_metric_landmarks_.colwise().homogeneous())
              .row(2);
    }
    float second_iteration_scale;
    if (EstimateScale(intermediate_landmarks, &second_iteration_scale) != GustoStatus::ERR_OK) {
        std::cerr << "Failed to estimate first iteration scale!" << std::endl;
        return GustoStatus::ERR_GENERAL_ERROR;
    }

    // Use the total scale to unproject the screen landmarks.
    const float total_scale = first_iteration_scale * second_iteration_scale;
    MoveAndRescaleZ(pcf, depth_offset, total_scale, screen_landmarks);
    UnprojectXY(pcf, screen_landmarks);
    ChangeHandedness(screen_landmarks);

    // At this point, screen landmarks are converted into metric landmarks.
    Eigen::Matrix3Xf& metric_landmarks = screen_landmarks;

    if (procrustes_solver_->SolveWeightedOrthogonalProblem(canonical_metric_landmarks_, metric_landmarks, landmark_weights_, pose_transform_mat) != GustoStatus::ERR_OK) {
        std::cerr << "Failed to estimate pose transform matrix!" << std::endl;
        return GustoStatus::ERR_GENERAL_ERROR;
    }

    // We do not use face detection result to predict the pose here, so deleted it.

    // Multiply each of the metric landmarks by the inverse pose
    // transformation matrix to align the runtime metric face landmarks with
    // the canonical metric face landmarks.
    metric_landmarks = (pose_transform_mat.inverse() *
                        metric_landmarks.colwise().homogeneous())
                           .topRows(3);

    ConvertEigenMatrixToLandmarkList(metric_landmarks, metric_landmark_list);

    return GustoStatus::ERR_OK;
  }

 private:
  void ProjectXY(const PerspectiveCameraFrustum& pcf,
                 Eigen::Matrix3Xf& landmarks) const {
    float x_scale = pcf.right - pcf.left;
    float y_scale = pcf.top - pcf.bottom;
    float x_translation = pcf.left;
    float y_translation = pcf.bottom;

    if (origin_point_location_ == OriginPointLocation::TOP_LEFT_CORNER) {
      landmarks.row(1) = 1.f - landmarks.row(1).array();
    }

    landmarks =
        landmarks.array().colwise() * Eigen::Array3f(x_scale, y_scale, x_scale);
    landmarks.colwise() += Eigen::Vector3f(x_translation, y_translation, 0.f);
  }

  GUSTO_RET EstimateScale(Eigen::Matrix3Xf& landmarks, float* ret) const {
    Eigen::Matrix4f transform_mat;
    if (procrustes_solver_->SolveWeightedOrthogonalProblem(
            canonical_metric_landmarks_, landmarks, landmark_weights_,
            transform_mat) != GustoStatus::ERR_OK) {
        std::cerr << "Failed to estimate canonical-to-runtime landmark set transform!" << std::endl;
        return GustoStatus::ERR_GENERAL_ERROR;
    }
    *ret = transform_mat.col(0).norm();
    return GustoStatus::ERR_OK;
    // return transform_mat.col(0).norm();
  }

  static void MoveAndRescaleZ(const PerspectiveCameraFrustum& pcf,
                              float depth_offset, float scale,
                              Eigen::Matrix3Xf& landmarks) {
    landmarks.row(2) =
        (landmarks.array().row(2) - depth_offset + pcf.near) / scale;
  }

  static void UnprojectXY(const PerspectiveCameraFrustum& pcf,
                          Eigen::Matrix3Xf& landmarks) {
    landmarks.row(0) =
        landmarks.row(0).cwiseProduct(landmarks.row(2)) / pcf.near;
    landmarks.row(1) =
        landmarks.row(1).cwiseProduct(landmarks.row(2)) / pcf.near;
  }

  static void ChangeHandedness(Eigen::Matrix3Xf& landmarks) {
    landmarks.row(2) *= -1.f;
  }

  static void ConvertLandmarkListToEigenMatrix(
      const NormalizedLandmarkList& landmark_list,
      Eigen::Matrix3Xf& eigen_matrix) {
    eigen_matrix = Eigen::Matrix3Xf(3, landmark_list.landmark_size());
    for (int i = 0; i < landmark_list.landmark_size(); ++i) {
      const auto& landmark = landmark_list.landmark[i];
      eigen_matrix(0, i) = landmark.x;
      eigen_matrix(1, i) = landmark.y;
      eigen_matrix(2, i) = landmark.z;
    }
  }

  static void ConvertEigenMatrixToLandmarkList(
      const Eigen::Matrix3Xf& eigen_matrix, LandmarkList& landmark_list) {
    landmark_list.Clear();

    for (int i = 0; i < eigen_matrix.cols(); ++i) {
      const Landmark landmark{x: eigen_matrix(0, i), y: eigen_matrix(1, i), z: eigen_matrix(2, i)};
      landmark_list.add_landmark(landmark);
      // auto& landmark = *landmark_list.add_landmark();
      // landmark.set_x(eigen_matrix(0, i));
      // landmark.set_y(eigen_matrix(1, i));
      // landmark.set_z(eigen_matrix(2, i));
    }
  }

  const OriginPointLocation origin_point_location_;
  const InputSource input_source_;
  Eigen::Matrix3Xf canonical_metric_landmarks_;
  Eigen::VectorXf landmark_weights_;

  std::unique_ptr<ProcrustesSolver> procrustes_solver_;
};




class GeometryPipelineImpl : public GeometryPipeline {
 public:
    GeometryPipelineImpl(
        const PerspectiveCamera& perspective_camera,  //
        const Mesh3d& canonical_mesh,                 //
        uint32_t canonical_mesh_vertex_size,          //
        uint32_t canonical_mesh_num_vertices,
        uint32_t canonical_mesh_vertex_position_offset,
        std::unique_ptr<ScreenToMetricSpaceConverter> space_converter)
        : perspective_camera_(perspective_camera),
        canonical_mesh_(canonical_mesh),
        canonical_mesh_vertex_size_(canonical_mesh_vertex_size),
        canonical_mesh_num_vertices_(canonical_mesh_num_vertices),
        canonical_mesh_vertex_position_offset_(canonical_mesh_vertex_position_offset),
        space_converter_(std::move(space_converter)) {}

  std::pair<std::vector<FaceGeometry>, GUSTO_RET> EstimateFaceGeometry(
  // std::optional<std::vector<FaceGeometry>> EstimateFaceGeometry(
      const std::vector<NormalizedLandmarkList>& multi_face_landmarks,
      int frame_width, 
      int frame_height) const override {
    
    if (ValidateFrameDimensions(frame_width, frame_height) != GustoStatus::ERR_OK) {
        std::cerr << "Invalid frame dimensions!" << std::endl;
        return std::make_pair(std::vector<FaceGeometry>(), GustoStatus::ERR_GENERAL_INVALID_PARAMETER);
        // return std::nullptr;
    }
    // Create a perspective camera frustum to be shared for geometry estimation
    // per each face.
    PerspectiveCameraFrustum pcf(perspective_camera_, frame_width,
                                 frame_height);

    std::vector<FaceGeometry> multi_face_geometry;
    // From this point, the meaning of "face landmarks" is clarified further as
    // "screen face landmarks". This is done do distinguish from "metric face
    // landmarks" that are derived during the face geometry estimation process.
    for (const NormalizedLandmarkList& screen_face_landmarks : multi_face_landmarks) {
        // Having a too compact screen landmark list will result in numerical
        // instabilities, therefore such faces are filtered.
        if (IsScreenLandmarkListTooCompact(screen_face_landmarks)) {
            std::cerr << "Screen landmark list is too compact!" << std::endl;
            continue;
        }
        // Convert the screen landmarks into the metric landmarks and get the pose
        // transformation matrix.
        LandmarkList metric_face_landmarks;
        Eigen::Matrix4f pose_transform_mat;
        if (space_converter_->Convert(screen_face_landmarks, pcf, metric_face_landmarks, pose_transform_mat) != GustoStatus::ERR_OK) {
            std::cerr << "Failed to convert landmarks from the screen to the metric space!" << std::endl;
            return std::make_pair(multi_face_geometry, GustoStatus::ERR_GENERAL_INVALID_PARAMETER);
            // return std::nullptr;
        }
        // [Sombra] -> I think it's for protobuf to send the pose matrix back
        // Pack geometry data for this face.
        // FaceGeometry face_geometry;
        // Mesh3d* mutable_mesh = face_geometry.mutable_mesh();
        // // Copy the canonical face mesh as the face geometry mesh.
        // mutable_mesh->CopyFrom(canonical_mesh_);
        // // Replace XYZ vertex mesh coodinates with the metric landmark positions.
        // for (int i = 0; i < canonical_mesh_num_vertices_; ++i) {
        //     uint32_t vertex_buffer_offset = canonical_mesh_vertex_size_ * i + canonical_mesh_vertex_position_offset_;

        //     mutable_mesh->set_vertex_buffer(vertex_buffer_offset, metric_face_landmarks.landmark(i).x());
        //     mutable_mesh->set_vertex_buffer(vertex_buffer_offset + 1, metric_face_landmarks.landmark(i).y());
        //     mutable_mesh->set_vertex_buffer(vertex_buffer_offset + 2, metric_face_landmarks.landmark(i).z());
        // }
        // // Populate the face pose transformation matrix.
        // mediapipe::MatrixDataProtoFromMatrix(pose_transform_mat, face_geometry.mutable_pose_transform_matrix());
        MatrixData MatrixData{4, 4, Layout::COLUMN_MAJOR};
        FaceGeometry _face_geometry{canonical_mesh_, MatrixData};
        // _face_geometry.mesh = canonical_mesh_;
        // _face_geometry.pose_transform_matrix = MatrixData{4, 4, Layout::COLUMN_MAJOR};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                _face_geometry.pose_transform_matrix.at(i, j) = pose_transform_mat(i, j);
            }
        }
        multi_face_geometry.push_back(_face_geometry);
    }

    return std::make_pair(multi_face_geometry, GustoStatus::ERR_OK); 
  }

 private:
  static bool IsScreenLandmarkListTooCompact(
      const NormalizedLandmarkList& screen_landmarks) {
    float mean_x = 0.f;
    float mean_y = 0.f;
    for (int i = 0; i < screen_landmarks.landmark_size(); ++i) {
      const auto& landmark = screen_landmarks.landmark[i];
      mean_x += (landmark.x - mean_x) / static_cast<float>(i + 1);
      mean_y += (landmark.y - mean_y) / static_cast<float>(i + 1);
    }

    float max_sq_dist = 0.f;
    for (const auto& landmark : screen_landmarks.landmark) {
      const float d_x = landmark.x - mean_x;
      const float d_y = landmark.y - mean_y;
      max_sq_dist = std::max(max_sq_dist, d_x * d_x + d_y * d_y);
    }

    static constexpr float kIsScreenLandmarkListTooCompactThreshold = 1e-3f;
    return std::sqrt(max_sq_dist) <= kIsScreenLandmarkListTooCompactThreshold;
  }

  const PerspectiveCamera perspective_camera_;
  const Mesh3d canonical_mesh_;
  const uint32_t canonical_mesh_vertex_size_;
  const uint32_t canonical_mesh_num_vertices_;
  const uint32_t canonical_mesh_vertex_position_offset_;

  std::unique_ptr<ScreenToMetricSpaceConverter> space_converter_;
};

// std::optional<std::unique_ptr<GeometryPipeline>> CreateGeometryPipeline(
std::pair<std::unique_ptr<GeometryPipeline>, GUSTO_RET> CreateGeometryPipeline(const Environment& environment, const GeometryPipelineMetadata& metadata) {
    if (ValidateEnvironment(environment) != GustoStatus::ERR_OK) {
        // return std::nullptr;
        return std::make_pair(nullptr, GustoStatus::ERR_GENERAL_INVALID_PARAMETER);
    }
    if (ValidateGeometryPipelineMetadata(metadata) != GustoStatus::ERR_OK) {
        // return std::nullptr;
        return std::make_pair(nullptr, GustoStatus::ERR_GENERAL_INVALID_PARAMETER);
    }
    const auto& canonical_mesh = metadata.canonical_mesh;

    // [Sombra] -> Don't need check canonical mesh here, it's hardcoded for us
    // if (ValidateCanonicalMesh(canonical_mesh) != GustoStatus::ERR_OK) {
    //     // return std::nullptr;
    //     return std::make_pair(nullptr, GustoStatus::ERR_GENERAL_INVALID_PARAMETER);
    // }
    // RET_CHECK(HasVertexComponent(canonical_mesh.vertex_type(),
    //                            VertexComponent::POSITION))
    //   << "Canonical face mesh must have the `POSITION` vertex component!";
    // RET_CHECK(HasVertexComponent(canonical_mesh.vertex_type(),
    //                            VertexComponent::TEX_COORD))
    //   << "Canonical face mesh must have the `TEX_COORD` vertex component!";

    uint32_t canonical_mesh_vertex_size = canonical_mesh.canonical_mesh_vertex_size;
    uint32_t canonical_mesh_num_vertices = canonical_mesh.canonical_mesh_num_vertices;
    uint32_t canonical_mesh_vertex_position_offset = canonical_mesh.canonical_mesh_vertex_position_offset;

    // Put the Procrustes landmark basis into Eigen matrices for an easier access.
    Eigen::Matrix3Xf canonical_metric_landmarks = Eigen::Matrix3Xf::Zero(3, canonical_mesh_num_vertices);
    Eigen::VectorXf landmark_weights = Eigen::VectorXf::Zero(canonical_mesh_num_vertices);

    for (int i = 0; i < canonical_mesh.canonical_mesh_num_vertices; ++i) {
        uint32_t vertex_buffer_offset =
            canonical_mesh.canonical_mesh_vertex_size * i + canonical_mesh.canonical_mesh_vertex_position_offset;
        canonical_metric_landmarks(0, i) = canonical_mesh.vertex_buffer[vertex_buffer_offset];
        canonical_metric_landmarks(1, i) = canonical_mesh.vertex_buffer[vertex_buffer_offset + 1];
        canonical_metric_landmarks(2, i) = canonical_mesh.vertex_buffer[vertex_buffer_offset + 2];
    }

    // # for example: 
    // # procrustes_landmark_basis { landmark_id: 4 weight: 0.070909939706326 }
    // # procrustes_landmark_basis { landmark_id: 6 weight: 0.032100144773722 }
    // # procrustes_landmark_basis { landmark_id: 10 weight: 0.008446550928056 }
    // # procrustes_landmark_basis { landmark_id: 33 weight: 0.058724168688059 }
    for (const WeightedLandmarkRef& wlr : metadata.procrustes_landmark_basis) {
        uint32_t landmark_id = wlr.landmark_id;
        landmark_weights(landmark_id) = wlr.weight;
    }

    std::unique_ptr<GeometryPipeline> result = 
        std::make_unique<GeometryPipelineImpl>(
            environment.perspective_camera, canonical_mesh,
            canonical_mesh_vertex_size, canonical_mesh_num_vertices,
            canonical_mesh_vertex_position_offset,
            std::make_unique<ScreenToMetricSpaceConverter>(
                environment.origin_point_location,
                metadata.input_source == InputSource::DEFAULT
                    ? InputSource::FACE_LANDMARK_PIPELINE
                    : metadata.input_source,
                std::move(canonical_metric_landmarks),
                std::move(landmark_weights),
                CreateFloatPrecisionProcrustesSolver()));

  return std::make_pair(std::move(result), GustoStatus::ERR_OK);
}

}