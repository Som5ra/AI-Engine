#include "utils.h"

using json = nlohmann::json;


enum InputSource {
    DEFAULT,
    FACE_LANDMARK_PIPELINE,
    FACE_DETECTION_PIPELINE,
};

enum OriginPointLocation {
    TOP_LEFT_CORNER,
    BOTTOM_LEFT_CORNER,
};

NLOHMANN_JSON_SERIALIZE_ENUM( InputSource, {
    {DEFAULT, nullptr},
    {FACE_LANDMARK_PIPELINE, "FACE_LANDMARK_PIPELINE"},
    {FACE_DETECTION_PIPELINE, "FACE_DETECTION_PIPELINE"},
})

struct _procrustes_landmark_basis
{
    int landmark_id;
    float weight;
};

struct _canonical_mesh
{
    std::vector<float> vertices;
    std::vector<int> faces;
    std::vector<float> vertex_buffer;
    std::vector<int> index_buffer;
};

class landmark_metadata
{
    public:
        // std::string input_source;
        InputSource input_source;
        std::vector<_procrustes_landmark_basis> procrustes_landmark_basis;
        _canonical_mesh canonical_mesh;
        void serialize_json(const std::string& filename);
};

void landmark_metadata::serialize_json(const std::string& filename)
{
    GustoSerializer serializer;
    json data = serializer.load_json(filename);
    this->input_source = data.template get<InputSource>();
    assert(this->input_source == InputSource::FACE_LANDMARK_PIPELINE);
    for(auto& it : data["procrustes_landmark_basis"]) {
        _procrustes_landmark_basis tmp;
        tmp.landmark_id = it["landmark_id"].get<int>();
        tmp.weight = it["weight"].get<float>();
        this->procrustes_landmark_basis.push_back(tmp);
    }
    for(auto& it : data["canonical_mesh"]["vertex_buffer"]) {
        this->canonical_mesh.vertex_buffer.push_back(it.get<float>());
    }
    for(auto& it : data["canonical_mesh"]["index_buffer"]) {
        this->canonical_mesh.index_buffer.push_back(it.get<int>());
    }
}


int main() {
    const std::string filename = "/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/face_geometry/geometry_pipeline_metadata_including_iris_landmarks.json";
    landmark_metadata metadata;
    metadata.serialize_json(filename);
    std::cout << metadata.input_source << std::endl;
    std::cout << metadata.canonical_mesh.vertex_buffer.size() << std::endl;
    std::cout << metadata.canonical_mesh.index_buffer.size() << std::endl;
    return 0;
}