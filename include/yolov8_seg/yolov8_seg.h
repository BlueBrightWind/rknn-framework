#ifndef YOLOV8_SEG_DETECTOR
#define YOLOV8_SEG_DETECTOR

#include <base/model.h>

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class YOLOV8SEG : public BaseModel {
   private:
    vector<float> transform_matrix;
    vector<int> input_size;
    bool is_quant;

   private:
    void preprocess(Mat& image);
    void postprocess(vector<vector<float>>& boxes, vector<vector<float>>& mask, float conf_thresh, float iou_thresh);

   public:
    bool init(string path, rknn_core_mask core_mask);
    bool inputImage(Mat& image);
    pair<vector<vector<float>>, vector<vector<float>>> getResult(float conf_thresh = 0.2, float iou_thresh = 0.4);
};

#endif
