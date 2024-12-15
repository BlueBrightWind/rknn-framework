#include "yolo_world/clip_text.h"

bool CLIPTEXT::initModelInputs() {
    for (int i = 0; i < io_num.n_input; i++) {
        inputs[i].index = i;
        inputs[i].pass_through = false;
        inputs[i].type = RKNN_TENSOR_INT32;
        inputs[i].fmt = RKNN_TENSOR_UNDEFINED;
        inputs[i].size = input_attrs[i].size;
        input_buffers[i] = new char[input_attrs[i].size];
        inputs[i].buf = input_buffers[i];
    }
    return true;
}

void CLIPTEXT::preprocess(vector<string>& texts) {
}

void CLIPTEXT::postprocess(vector<vector<float>>& result, float conf_thresh, float iou_thresh) {
}
