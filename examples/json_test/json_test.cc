#include "utils.h"

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    const std::string filename = argc > 1
        ? argv[1]
        : "tools/face_geometry/geometry_pipeline_metadata_including_iris_landmarks.json";

    try {
        std::ifstream input(filename);
        if (!input.is_open()) {
            std::cerr << "Unable to open metadata: " << filename << std::endl;
            return 1;
        }

        const nlohmann::json metadata = nlohmann::json::parse(input);
        const std::string input_source =
            metadata.at("input_source").get<std::string>();
        const auto& basis = metadata.at("procrustes_landmark_basis");
        const auto& mesh = metadata.at("canonical_mesh");
        const auto& vertices = mesh.at("vertex_buffer");
        const auto& indices = mesh.at("index_buffer");

        if (input_source != "FACE_LANDMARK_PIPELINE" || basis.empty() ||
            vertices.empty() || indices.empty() || vertices.size() % 5 != 0) {
            std::cerr << "Metadata schema or canonical mesh is invalid"
                      << std::endl;
            return 1;
        }

        std::cout << "input source: " << input_source << '\n'
                  << "basis entries: " << basis.size() << '\n'
                  << "mesh vertices: " << vertices.size() / 5 << '\n'
                  << "mesh indices: " << indices.size() << std::endl;
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Metadata validation failed: " << exception.what()
                  << std::endl;
        return 1;
    }
}
