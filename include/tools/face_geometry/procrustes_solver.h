#include <memory>
#include "Eigen/Dense"
extern "C" {

namespace face_geometry {

// Encapsulates a stateless solver for the Weighted Extended Orthogonal
// Procrustes (WEOP) Problem, as defined in Section 2.4 of
// https://doi.org/10.3929/ethz-a-004656648.
//
// Given the source and the target point clouds, the algorithm estimates
// a 4x4 transformation matrix featuring the following semantic components:
//
//   * Uniform scale
//   * Rotation
//   * Translation
//
// The matrix maps the source point cloud into the target point cloud minimizing
// the Mean Squared Error.
class ProcrustesSolver {
 public:
  virtual ~ProcrustesSolver() = default;

  // Solves the Weighted Extended Orthogonal Procrustes (WEOP) Problem.
  //
  // All `source_points`, `target_points` and `point_weights` must define the
  // same number of points. Elements of `point_weights` must be non-negative.
  //
  // A too small diameter of either of the point clouds will likely lead to
  // numerical instabilities and failure to estimate the transformation.
  //
  // A too small point cloud total weight will likely lead to numerical
  // instabilities and failure to estimate the transformation too.
  //
  // Small point coordinate deviation for either of the point cloud will likely
  // result in a failure as it will make the solution very unstable if possible.
  //
  // Note: the output `transform_mat` argument is used instead of `StatusOr<>`
  // return type in order to avoid Eigen memory alignment issues. Details:
  // https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
  virtual int SolveWeightedOrthogonalProblem(
      const Eigen::Matrix3Xf& source_points,  //
      const Eigen::Matrix3Xf& target_points,  //
      const Eigen::VectorXf& point_weights,   //
      Eigen::Matrix4f& transform_mat) const = 0;
};

std::unique_ptr<ProcrustesSolver> CreateFloatPrecisionProcrustesSolver();

}  // namespace mediapipe::face_geometry


}