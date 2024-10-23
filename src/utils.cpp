#include "utils.h"

extern "C"
{
    Net_config::Net_config(int inpHeight, int inpWidth,
                        float confThreshold, float nmsThreshold, 
                        const char* modelpath, const char* cls_names_path, int len_string
    ){
        this->confThreshold = confThreshold;
        this->nmsThreshold = nmsThreshold;

        this->modelpath.assign(modelpath, len_string);
        this->cls_names_path.assign(cls_names_path, len_string);
        ifstream ifs(this->cls_names_path.c_str());
        string line;
        while (getline(ifs, line)) this->class_names.push_back(line);
        this->num_class = class_names.size();
        this->inpHeight = inpHeight;
        this->inpWidth = inpWidth;
    }
}