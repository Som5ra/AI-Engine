#ifndef GUSTO_FACE_GEOMETRY_H
#define GUSTO_FACE_GEOMETRY_H
#include "utils.h"

namespace gusto_face_geometry {

    #define GUSTO_RET int
    using json = nlohmann::json;



    enum InputSource {
        DEFAULT,
        FACE_LANDMARK_PIPELINE,
        FACE_DETECTION_PIPELINE,
    };
    NLOHMANN_JSON_SERIALIZE_ENUM( InputSource, {
        {DEFAULT, nullptr},
        {FACE_LANDMARK_PIPELINE, "FACE_LANDMARK_PIPELINE"},
        {FACE_DETECTION_PIPELINE, "FACE_DETECTION_PIPELINE"},
    })


    struct WeightedLandmarkRef
    {
        int landmark_id;
        float weight;
    };

    struct Mesh3d
    {
        std::vector<float> vertices;
        std::vector<int> faces;
        std::vector<float> vertex_buffer;
        std::vector<int> index_buffer;
        uint32_t canonical_mesh_num_vertices = 478; // actually 478 points
        uint32_t canonical_mesh_vertex_size = 5; // 3 vertices (xyz) and 2 texture coordinates (uv)
        uint32_t canonical_mesh_vertex_position_offset = 0; // the position of the vertices in the vertex buffer
        uint32_t primitive_type = 3; // 3 vertices per face
        int vertex_buffer_size() const { return vertex_buffer.size(); }
        int index_buffer_size() const { return index_buffer.size(); }
    };

    class GeometryPipelineMetadata
    {
        public:
            // std::string input_source;
            InputSource input_source;
            std::vector<WeightedLandmarkRef> procrustes_landmark_basis;
            Mesh3d canonical_mesh;
            GUSTO_RET serialize_json(const std::string& filename);
            GUSTO_RET GUSTO_CHECK_CANONICAL_MESH();
            int procrustes_landmark_basis_size() const { return procrustes_landmark_basis.size(); }
    };


    struct PerspectiveCamera {
        float vertical_fov_degrees() const { return vertical_fov_degrees_; }
        float near() const { return near_; }
        float far() const { return far_; }

        float vertical_fov_degrees_;
        float near_;
        float far_;
    };

    enum OriginPointLocation {
        TOP_LEFT_CORNER,
        BOTTOM_LEFT_CORNER,
    };

    struct Environment {
        OriginPointLocation origin_point_location;
        PerspectiveCamera perspective_camera;
        Environment(OriginPointLocation origin, const PerspectiveCamera& camera) : origin_point_location(origin), perspective_camera(camera) {}
    };

    struct PerspectiveCameraFrustum {
        // NOTE: all arguments must be validated prior to calling this constructor.
        PerspectiveCameraFrustum(const PerspectiveCamera& perspective_camera,
                            int frame_width, int frame_height) {
            static constexpr float kDegreesToRadians = 3.14159265358979323846f / 180.f;

            const float height_at_near =
                2.f * perspective_camera.near() *
                std::tan(0.5f * kDegreesToRadians *
                        perspective_camera.vertical_fov_degrees());

            const float width_at_near = frame_width * height_at_near / frame_height;

            left = -0.5f * width_at_near;
            right = 0.5f * width_at_near;
            bottom = -0.5f * height_at_near;
            top = 0.5f * height_at_near;
            near = perspective_camera.near();
            far = perspective_camera.far();
        }
        float left;
        float right;
        float bottom;
        float top;
        float near;
        float far;
    };

    enum Layout {
        COLUMN_MAJOR,
        ROW_MAJOR
    };

    struct MatrixData {
        uint32_t rows;
        uint32_t cols;
        std::vector<float> packed_data;
        Layout layout;
        int packed_data_size() const { return rows * cols; }
        // Constructor to initialize the vector with the correct size
        MatrixData(uint32_t r, uint32_t c, Layout l)
            : rows(r), cols(c), packed_data(r * c), layout(l) {}

        // Access element at (i, j)
        float& at(uint32_t i, uint32_t j) {
            return packed_data[i * cols + j];
        }

        const float& at(uint32_t i, uint32_t j) const {
            return packed_data[i * cols + j];
        }
    };

    struct FaceGeometry
    {
        Mesh3d mesh;
        MatrixData pose_transform_matrix;
    };


    struct Landmark
    {
        float x;
        float y;
        float z;
        // Landmark visibility. Should stay unset if not supported.
        // Float score of whether landmark is visible or occluded by other objects.
        // Landmark considered as invisible also if it is not present on the screen
        // (out of scene bounds). Depending on the model, visibility value is either a
        // sigmoid or an argument of sigmoid.
        float visibility;

        // Landmark presence. Should stay unset if not supported.
        // Float score of whether landmark is present on the scene (located within
        // scene bounds). Depending on the model, presence value is either a result of
        // sigmoid or an argument of sigmoid function to get landmark presence
        // probability.
        float presence;
    };

    struct LandmarkList
    {
        std::vector<Landmark> landmark;
        int landmark_size() const { return landmark.size(); }
        void add_landmark(const Landmark& l) { landmark.push_back(l); }
        void Clear() { landmark.clear(); }
    };
    struct LandmarkListCollection
    {
        std::vector<LandmarkList> landmark_list;
        int landmark_list_size() const { return landmark_list.size(); }
    };

    // A normalized version of above Landmark proto. All coordinates should be
    // within [0, 1].
    struct NormalizedLandmark
    {
        float x;
        float y;
        float z;
        // Landmark visibility. Should stay unset if not supported.
        // Float score of whether landmark is visible or occluded by other objects.
        // Landmark considered as invisible also if it is not present on the screen
        // (out of scene bounds). Depending on the model, visibility value is either a
        // sigmoid or an argument of sigmoid.
        float visibility;

        // Landmark presence. Should stay unset if not supported.
        // Float score of whether landmark is present on the scene (located within
        // scene bounds). Depending on the model, presence value is either a result of
        // sigmoid or an argument of sigmoid function to get landmark presence
        // probability.
        float presence;
    };

    // Group of NormalizedLandmark protos.
    struct NormalizedLandmarkList
    {
        std::vector<NormalizedLandmark> landmark;
        int landmark_size() const { return landmark.size(); }
    };

    struct NormalizedLandmarkListCollection
    {
        std::vector<NormalizedLandmarkList> landmark_list;
        int landmark_list_size() const { return landmark_list.size(); }
    };

    
    // Validates `perspective_camera`.
    //
    // Near Z must be greater than 0 with a margin of `1e-9`.
    // Far Z must be greater than Near Z with a margin of `1e-9`.
    // Vertical FOV must be in range (0, 180) with a margin of `1e-9` on the range
    // edges.
    GUSTO_RET ValidatePerspectiveCamera(const PerspectiveCamera& perspective_camera);

    // Validates `environment`.
    //
    // Environment's perspective camera must be valid.
    GUSTO_RET ValidateEnvironment(const Environment& environment);

    // Validates `mesh_3d`.
    //
    // Mesh vertex buffer size must a multiple of the vertex size.
    // Mesh index buffer size must a multiple of the primitive size.
    // All mesh indices must reference an existing mesh vertex.
    GUSTO_RET ValidateMesh3d(const Mesh3d& mesh_3d);

    // Validates `face_geometry`.
    //
    // Face mesh must be valid.
    // Face pose transformation matrix must be a 4x4 matrix.
    GUSTO_RET ValidateFaceGeometry(const FaceGeometry& face_geometry);

    // Validates `metadata`.
    //
    // Canonical face mesh must be valid.
    // Procrustes landmark basis must be non-empty.
    // All Procrustes basis indices must reference an existing canonical mesh
    // vertex.
    // All Procrustes basis landmarks must have a non-negative weight.
    GUSTO_RET ValidateGeometryPipelineMetadata(
        const GeometryPipelineMetadata& metadata);

    // Validates frame dimensions.
    //
    // Both frame width and frame height must be positive.
    GUSTO_RET ValidateFrameDimensions(int frame_width, int frame_height);


}

#endif // GUSTO_FACE_GEOMETRY_H