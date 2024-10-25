#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <opencv2/opencv.hpp> 
#include <nlohmann/json.hpp>
#include "utils.h"

extern "C"
{
    using namespace std;
    using namespace cv;
    using namespace dnn;


    class GenericDetector
    {
        public:
            GenericDetector();
            void detect(cv::Mat& frame, float* ret_bboxes, float* ret_confidences, int* ret_classIds, int* ret_len);
            int compile(int inpHeight, int inpWidth,
                        float confThreshold, float nmsThreshold, 
                        const char* modelpath, const char* cls_names_path, int len_string);
            Net_config* config;

        private:

            cv::dnn::Net* net;
            // disabled for android testing
            // void drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, int classid);
    };
    
    GenericDetector::GenericDetector()
    {
        cv::setNumThreads(std::max(1, omp_get_num_procs() / 2));
        // this->net = new cv::dnn::Net();
        // this->config = new Net_config();
    }

    int GenericDetector::compile(int inpHeight, int inpWidth,
                        float confThreshold, float nmsThreshold, 
                        const char* modelpath, const char* cls_names_path, int len_string = 1024
    ){
        // if succeed return 0 else 1
        try {
            this->config = new Net_config(inpHeight, inpWidth, confThreshold, nmsThreshold, modelpath, cls_names_path, 1024);
            // cv::dnn::Net net = cv::dnn::readNetFromONNX(this->config->modelpath);
            // this->net = &net;
            this->net = new cv::dnn::Net(cv::dnn::readNetFromONNX(this->config->modelpath));
            this->net->setPreferableBackend(DNN_BACKEND_OPENCV);
            this->net->setPreferableTarget(DNN_TARGET_CPU);
            // this->net = &;
            ifstream ifs(this->config->cls_names_path.c_str());
            string line;

        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            return 1;
        }
        return 0;
    }


    // disabled for android testing
    // void GenericDetectorV7::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid)   // Draw the predicted bounding box
    // {
    //     //Draw a rectangle displaying the bounding box
    //     rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);
    //     //Get the label for the class name and its confidence
    //     string label = format("%.2f", conf);
    //     label = this->class_names[classid] + ":" + label;

    //     //Display the label at the top of the bounding box
    //     int baseLine;
    //     Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    //     top = max(top, labelSize.height);
    //     //rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
    //     putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
    // }


    void GenericDetector::detect(Mat& frame, float* ret_bboxes, float* ret_confidences, int* ret_classIds, int* ret_len)
    {

        auto time_1 = std::chrono::high_resolution_clock::now();

        Mat blob = blobFromImage(frame, 1 / 255.0, Size(this->config->inpWidth, this->config->inpHeight), Scalar(0, 0, 0), true, false);
        this->net->setInput(blob);	
        vector<Mat> outs;

        auto time_2 = std::chrono::high_resolution_clock::now();

        std::cout << "preprocessing time: " << std::chrono::duration<double, std::milli>(time_2-time_1).count() << std::endl;

        this->net->forward(outs, this->net->getUnconnectedOutLayersNames());
        
        
        auto time_3 = std::chrono::high_resolution_clock::now();
        std::cout << "network time: " << std::chrono::duration<double, std::milli>(time_3-time_2).count() << std::endl;

        vector<float> confidences;
        vector<Rect> boxes;
        vector<int> classIds;
        // yolov7 post-processing
        if (outs.size() == 1)
        {
            int num_proposal = outs[0].size[0];
            int nout = outs[0].size[1];
            if (outs[0].dims > 2)
            {
                num_proposal = outs[0].size[1];
                nout = outs[0].size[2];
                outs[0] = outs[0].reshape(0, num_proposal);
            }

            /////generate proposals

            float ratioh = (float)frame.rows / this->config->inpHeight, ratiow = (float)frame.cols / this->config->inpWidth;
            int n = 0, row_ind = 0; ///cx,cy,w,h,box_score,class_score
            float* pdata = (float*)outs[0].data;
            for (n = 0; n < num_proposal; n++)   
            {
                float box_score = pdata[4];
                if (box_score > this->config->confThreshold)
                {
                    Mat scores = outs[0].row(row_ind).colRange(5, nout);
                    Point classIdPoint;
                    double max_class_socre;
                    // Get the value and location of the maximum score
                    minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
                    max_class_socre *= box_score;
                    if (max_class_socre > this->config->confThreshold)
                    {
                        const int class_idx = classIdPoint.x;
                        float cx = pdata[0] * ratiow;  ///cx
                        float cy = pdata[1] * ratioh;   ///cy
                        float w = pdata[2] * ratiow;   ///w
                        float h = pdata[3] * ratioh;  ///h

                        int left = int(cx - 0.5 * w);
                        int top = int(cy - 0.5 * h);

                        confidences.push_back((float)max_class_socre);
                        boxes.push_back(Rect(left, top, (int)(w), (int)(h)));
                        classIds.push_back(class_idx);
                    }
                }
                row_ind++;
                pdata += nout;
            }
        }else if (outs.size() == 2) 
        {
            // rtmdet post-processing
            // TO DO

            // Mat boxes = outs[0].reshape(1, {outs[0].size[1], outs[0].size[2]});
            // Mat scores = outs[1].reshape(1, {outs[1].size[1], outs[1].size[2]});
            // for (int i = 0; i < boxes.rows; ++i) {
            //     boxes.push_back(Rect(Point(boxes.at<float>(i, 0), boxes.at<float>(i, 1)),
            //                             Point(boxes.at<float>(i, 2), boxes.at<float>(i, 3))));
            // }

        }
        

        auto time_4 = std::chrono::high_resolution_clock::now();
        std::cout << "post-processing 1-period time: " << std::chrono::duration<double, std::milli>(time_4-time_3).count() << std::endl;
        // Perform non maximum suppression to eliminate redundant overlapping boxes with
        // lower confidences
        vector<int> indices;
        dnn::NMSBoxes(boxes, confidences, this->config->confThreshold, this->config->nmsThreshold, indices);

        *ret_len = indices.size();
        for (size_t i = 0; i < *ret_len; ++i)
        {
            int idx = indices[i];
            Rect box = boxes[idx];
            // this->drawPred(confidences[idx], box.x, box.y,
            //     box.x + box.width, box.y + box.height, frame, classIds[idx]);
            ret_bboxes[i * 4] = box.x;
            ret_bboxes[i * 4 + 1] = box.y;
            ret_bboxes[i * 4 + 2] = box.width;
            ret_bboxes[i * 4 + 3] = box.height;
            ret_confidences[i] = confidences[idx];
            ret_classIds[i] = classIds[idx];
        }
        auto time_5 = std::chrono::high_resolution_clock::now();
        std::cout << "post-processing 2-period time: " << std::chrono::duration<double, std::milli>(time_5-time_4).count() << std::endl;
    }
}