#ifndef CLIP_TEXT
#define CLIP_TEXT

#include <base/model.h>
#include <string>

using namespace std;

class CLIPTEXT : public BaseModel {
   private:
    bool initModelInputs();

   private:
    void preprocess(vector<string>& texts);
    void postprocess(vector<vector<float>>& result, float conf_thresh, float iou_thresh);

   public:
    bool init(string path, rknn_core_mask core_mask);
    bool inputImage(vector<string>& texts);
    vector<vector<float>> getResult(float conf_thresh = 0.2, float iou_thresh = 0.4);
};

#endif