#ifndef YOLOV7_DETECTOR
#define YOLOV7_DETECTOR

#include <model.h>

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class YOLOV7 : public BaseModel {
   private:
    vector<float> transform_matrix;
    bool is_quant;

   private:
    void preprocess(Mat& image);
    void postprocess(vector<vector<float>>& result, float conf_thresh, float iou_thresh);

   public:
    bool init(string path, rknn_core_mask core_mask);
    bool inputImage(Mat& image);
    vector<vector<float>> getResult(float conf_thresh = 0.2, float iou_thresh = 0.4);
};

#endif
