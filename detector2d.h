#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp> 
#include <nlohmann/json.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

extern "C"
{
    struct Net_config
    {
        float confThreshold; // Confidence threshold
        float nmsThreshold;  // Non-maximum suppression threshold
        std::string modelpath;
        std::string cls_names_path;
        int inpHeight;
        int inpWidth;
    };

    class YOLOV7
    {
        public:
            YOLOV7(Net_config config);
            void detect(cv::Mat& frame, float* ret_bboxes, float* ret_confidences, int* ret_classIds, int ret_len);
        private:
            int inpWidth;
            int inpHeight;
            std::vector<std::string> class_names;
            int num_class;

            float confThreshold;
            float nmsThreshold;
            cv::dnn::Net net;
            // disabled for android testing
            // void drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, int classid);
    };
    
    YOLOV7::YOLOV7(Net_config config)
    {
        this->confThreshold = config.confThreshold;
        this->nmsThreshold = config.nmsThreshold;

        this->net = readNetFromONNX(config.modelpath);
        ifstream ifs(config.cls_names_path.c_str());
        string line;
        while (getline(ifs, line)) this->class_names.push_back(line);
        this->num_class = class_names.size();
        this->inpHeight = config.inpHeight;
        this->inpWidth = config.inpWidth;
    }
    // disabled for android testing
    // void YOLOV7::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid)   // Draw the predicted bounding box
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

    void YOLOV7::detect(Mat& frame, float* ret_bboxes, float* ret_confidences, int* ret_classIds, int ret_len)
    {
        Mat blob = blobFromImage(frame, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
        this->net.setInput(blob);	
        vector<Mat> outs;
        this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
        int num_proposal = outs[0].size[0];
        int nout = outs[0].size[1];
        if (outs[0].dims > 2)
        {
            num_proposal = outs[0].size[1];
            nout = outs[0].size[2];
            outs[0] = outs[0].reshape(0, num_proposal);
        }
        /////generate proposals
        vector<float> confidences;
        vector<Rect> boxes;
        vector<int> classIds;
        float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
        int n = 0, row_ind = 0; ///cx,cy,w,h,box_score,class_score
        float* pdata = (float*)outs[0].data;
        for (n = 0; n < num_proposal; n++)   
        {
            float box_score = pdata[4];
            if (box_score > this->confThreshold)
            {
                Mat scores = outs[0].row(row_ind).colRange(5, nout);
                Point classIdPoint;
                double max_class_socre;
                // Get the value and location of the maximum score
                minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
                max_class_socre *= box_score;
                if (max_class_socre > this->confThreshold)
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

        // Perform non maximum suppression to eliminate redundant overlapping boxes with
        // lower confidences
        vector<int> indices;
        dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);

        ret_len = indices.size();
        for (size_t i = 0; i < ret_len; ++i)
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
    }
}