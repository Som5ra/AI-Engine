#ifndef GUSTO_GEOMETRY_PIPELINE_H_
#define GUSTO_GEOMETRY_PIPELINE_H_
#include "utils.h"
#include "tools/face_geometry/face_geometry.h"
namespace gusto_face_geometry {

// Encapsulates a stateless estimator of facial geometry in a Metric space based
// on the normalized face landmarks in the Screen space.
class GeometryPipeline {
 public:
  virtual ~GeometryPipeline() = default;

  // Estimates geometry data for multiple faces.
  //
  // Returns an error status if any of the passed arguments is invalid.
  //
  // The result includes face geometry data for a subset of the input faces,
  // however geometry data for some faces might be missing. This may happen if
  // it'd be unstable to estimate the facial geometry based on a corresponding
  // face landmark list for any reason (for example, if the landmark list is too
  // compact).
  //
  // Each face landmark list must have the same number of landmarks as was
  // passed upon initialization via the canonical face mesh (as a part of the
  // geometry pipeline metadata).
  //
  // Both `frame_width` and `frame_height` must be positive.
  virtual std::pair<std::vector<FaceGeometry>, GUSTO_RET> EstimateFaceGeometry(
//   virtual std::optional<std::vector<FaceGeometry>> EstimateFaceGeometry(
      const std::vector<NormalizedLandmarkList>& multi_face_landmarks,
      int frame_width, int frame_height) const = 0;
};

// Creates an instance of `GeometryPipeline`.
//
// Both `environment` and `metadata` must be valid (for details, please refer to
// the proto message definition comments and/or `validation_utils.h/cc`).
//
// Canonical face mesh (defined as a part of `metadata`) must have the
// `POSITION` and the `TEX_COORD` vertex components.
std::pair<std::unique_ptr<GeometryPipeline>, GUSTO_RET> CreateGeometryPipeline(
// std::optional<std::unique_ptr<GeometryPipeline>> CreateGeometryPipeline(
    const Environment& environment, const GeometryPipelineMetadata& metadata);

}  // namespace face_geometry

#endif  // GUSTO_GEOMETRY_PIPELINE_H_