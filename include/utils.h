#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>
#include <opencv2/opencv.hpp> 

extern "C"
{
    #define ERR_OK                                                            0x00000000

    // General
    #define ERR_GENERAL_ERROR                                                 0x76000000
    #define ERR_GENERAL_FILE_IO                         -(ERR_GENERAL_ERROR | 0x00000001)
    #define ERR_GENERAL_IMAGE_LOAD                      -(ERR_GENERAL_ERROR | 0x00000002)
    #define ERR_GENERAL_SERIALIZATION                   -(ERR_GENERAL_ERROR | 0x00000003)
    #define ERR_GENERAL_INVALID_PARAMETER               -(ERR_GENERAL_ERROR | 0x00000004)
    #define ERR_GENERAL_NOT_SUPPORT                     -(ERR_GENERAL_ERROR | 0x00000005)

    
    using namespace std;
    using namespace cv;
    using namespace dnn;


    class Net_config
    {
        public:
            Net_config();
            Net_config(int inpHeight, int inpWidth,
                        float confThreshold, float nmsThreshold, 
                        const char* modelpath, const char* cls_names_path, int len_string = 1024);
            Net_config(const Net_config& config);

            float confThreshold; // Confidence threshold
            float nmsThreshold;  // Non-maximum suppression threshold

            int inpHeight;
            int inpWidth;
            
            std::string cls_names_path;
            std::string modelpath;
            std::vector<std::string> class_names;
            int num_class;

    };
}