#include "utils.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

Net_config::Net_config(int inpHeight, int inpWidth,
                    float confThreshold, float nmsThreshold, 
                    const char* modelpath, const char* cls_names_path, int len_string
){
    this->confThreshold = confThreshold;
    this->nmsThreshold = nmsThreshold;

    this->modelpath.assign(modelpath, len_string);
    this->cls_names_path.assign(cls_names_path, len_string);
    std::ifstream ifs(this->cls_names_path.c_str());
    std::string line;
    while (getline(ifs, line)) this->class_names.push_back(line);
    this->num_class = class_names.size();
    this->inpHeight = inpHeight;
    this->inpWidth = inpWidth;
}

// nlohmann::json GustoSerializer::load_json(const char* filename){
//     std::ifstream ifs(filename);
//     if (!ifs.is_open()) {
//         std::cerr << "Error opening JSON file: " << filename << std::endl;
//         return nlohmann::json();
//     }
//     return nlohmann::json::parse(ifs);
//     // return nlohmann::json();
// }
// nlohmann::json GustoSerializer::load_json(const std::string& filename){
//     std::ifstream ifs(filename.c_str());
//     if (!ifs.is_open()) {
//         std::cerr << "Error opening JSON file: " << filename << std::endl;
//         return nlohmann::json();
//     }
//     return nlohmann::json::parse(ifs);
//     // return load_json(filename.c_str());
// }

