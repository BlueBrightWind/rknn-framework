#include <math.h>
#include <yolov8.h>

#include <functional>

using namespace std;

bool YOLOV8::init(string path, rknn_core_mask core_mask) {
    if (!BaseModel::init(path, core_mask))
        return false;
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8)
        this->is_quant = true;
    else
        this->is_quant = false;
    return true;
}

void YOLOV8::preprocess(Mat& data) {
    int modelHeight = input_attrs[0].dims[1];
    int modelWidth = input_attrs[0].dims[2];
    int dataHeight = data.rows;
    int dataWidth = data.cols;
    float scale = min((float)modelHeight / dataHeight, (float)modelWidth / dataWidth);
    int dx = modelWidth - dataWidth * scale;
    int dy = modelHeight - dataHeight * scale;
    transform_matrix.resize(3);
    transform_matrix[0] = dx / 2;
    transform_matrix[1] = dy / 2;
    transform_matrix[2] = scale;
    resize(data, data, Size(dataWidth * scale, dataHeight * scale));
    copyMakeBorder(data, data, dy / 2, dy - dy / 2, dx / 2, dx - dx / 2, BORDER_CONSTANT);
}

void YOLOV8::postprocess(vector<vector<float>>& result, float conf_thresh, float iou_thresh) {
    function<int32_t(float, float, float)> clipValue = [&](float val, float min, float max) {
        float f = val <= min ? min : (val >= max ? max : val);
        return f;
    };

    function<int8_t(float, int32_t, float)> qntFp32ToAffine = [&](float f32, int32_t zp, float scale) {
        float dst_val = (f32 / scale) + zp;
        int8_t res = (int8_t)clipValue(dst_val, -128, 127);
        return res;
    };

    function<float(int8_t, int32_t, float)> deqntAffineToFp32 = [&](int8_t qnt, int32_t zp, float scale) {
        return ((float)qnt - (float)zp) * scale;
    };

    function<vector<float>(vector<float>)> computeDfl = [&](vector<float> tensor) {
        vector<float> position(4);
        int length = tensor.size() / 4;
        for (int i = 0; i < 4; i++) {
            vector<float> exp_t(length);
            float exp_sum = 0;
            float acc_sum = 0;
            for (int j = 0; j < length; j++) {
                exp_t[j] = exp(tensor[i * length + j]);
                exp_sum += exp_t[j];
            }
            for (int j = 0; j < length; j++) {
                acc_sum += exp_t[j] / exp_sum * j;
            }
            position[i] = acc_sum;
        }
        return position;
    };

    function<float(const vector<float>&, const vector<float>&)> computeIou = [&](const vector<float>& a, const vector<float>& b) {
        float interLeft = max(a[0], b[0]);
        float interTop = max(a[1], b[1]);
        float interRight = min(a[2], b[2]);
        float interBottom = min(a[3], b[3]);
        float interWidth = interRight - interLeft;
        float interHeight = interBottom - interTop;
        if (interWidth <= 0 || interHeight <= 0)
            return 0.0f;
        float interArea = interWidth * interHeight;
        float areaA = (a[2] - a[0]) * (a[3] - a[1]);
        float areaB = (b[2] - b[0]) * (b[3] - b[1]);
        return interArea / (areaA + areaB - interArea);
    };

    function<void(vector<vector<float>>&, float)> nms = [&](vector<vector<float>>& detections, float iou_thresh) {
        sort(detections.begin(), detections.end(),
             [&](const vector<float>& a, const vector<float>& b) -> bool {
                 return a[5] > b[5];
             });

        unordered_map<int, vector<vector<float>>> class_map;
        for (const auto& det : detections) {
            int class_id = (int)det[4];
            class_map[class_id].push_back(det);
        }

        detections.clear();

        for (auto& [class_id, class_dets] : class_map) {
            vector<vector<float>>& current_dets = class_dets;  // 复制以便修改
            vector<vector<float>> selected;

            while (!current_dets.empty()) {
                vector<float> current_box = current_dets.front();
                selected.push_back(current_box);
                current_dets.erase(current_dets.begin());
                current_dets.erase(
                    remove_if(current_dets.begin(), current_dets.end(),
                              [&](const vector<float>& det) -> bool {
                                  return computeIou(current_box, det) > iou_thresh;
                              }),
                    current_dets.end());
            }
            detections.insert(detections.end(), selected.begin(), selected.end());
        }
    };

    int height = input_attrs[0].dims[1];
    int width = input_attrs[0].dims[2];

    if (this->is_quant) {
        for (int i = 0; i < io_num.n_output / 3; i++) {
            int position_index = i * 3 + 0;
            int conf_index = i * 3 + 1;
            int box_index = i * 3 + 2;
            int8_t* position = (int8_t*)output_buffers[position_index];
            int8_t* conf = (int8_t*)output_buffers[conf_index];
            int8_t* box = (int8_t*)output_buffers[box_index];
            int position_zp = output_attrs[position_index].zp;
            int conf_zp = output_attrs[conf_index].zp;
            int box_zp = output_attrs[box_index].zp;
            float position_scale = output_attrs[position_index].scale;
            float conf_scale = output_attrs[conf_index].scale;
            float box_scale = output_attrs[box_index].scale;

            int grid_h = output_attrs[box_index].dims[2];
            int grid_w = output_attrs[box_index].dims[3];
            int grid_length = grid_h * grid_w;

            int class_num = output_attrs[conf_index].dims[1];
            int stride = sqrt(width * height / grid_w / grid_h);

            int8_t conf_thresh_i8 = qntFp32ToAffine(conf_thresh, conf_zp, conf_scale);
            int8_t box_thresh_i8 = qntFp32ToAffine(conf_thresh, box_zp, box_scale);

            for (int gh = 0; gh < grid_h; gh++) {
                for (int gw = 0; gw < grid_w; gw++) {
                    int index = gh * grid_w + gw;
                    if (box[index] < box_thresh_i8)
                        continue;
                    int8_t max_class_prob = conf[index];
                    int max_class_id = 0;
                    for (int k = 1; k < class_num; k++) {
                        int8_t prob = conf[k * grid_length + index];
                        if (prob > max_class_prob) {
                            max_class_id = k;
                            max_class_prob = prob;
                        }
                    }
                    if (max_class_prob < conf_thresh_i8)
                        continue;
                    int position_tensor_length = output_attrs[position_index].dims[1];
                    vector<float> position_tensor(position_tensor_length);
                    for (int k = 0; k < position_tensor_length; k++)
                        position_tensor[k] = deqntAffineToFp32(position[index + k * grid_length], position_zp, position_scale);
                    vector<float> position = computeDfl(position_tensor);
                    float x0 = (-position[0] + gw + 0.5) * stride;
                    float y0 = (-position[1] + gh + 0.5) * stride;
                    float x1 = (position[2] + gw + 0.5) * stride;
                    float y1 = (position[3] + gh + 0.5) * stride;
                    float class_id = max_class_id;
                    float conf = deqntAffineToFp32(max_class_prob, conf_zp, conf_scale);
                    result.push_back({x0, y0, x1, y1, class_id, conf});
                }
            }
        }
    } else {
        for (int i = 0; i < io_num.n_output / 3; i++) {
            int position_index = i * 3 + 0;
            int conf_index = i * 3 + 1;
            int box_index = i * 3 + 2;
            float16_t* position = (float16_t*)output_buffers[position_index];
            float16_t* conf = (float16_t*)output_buffers[conf_index];
            float16_t* box = (float16_t*)output_buffers[box_index];

            int grid_h = output_attrs[box_index].dims[2];
            int grid_w = output_attrs[box_index].dims[3];
            int grid_length = grid_h * grid_w;

            int class_num = output_attrs[conf_index].dims[1];
            int stride = sqrt(width * height / grid_w / grid_h);

            for (int gh = 0; gh < grid_h; gh++) {
                for (int gw = 0; gw < grid_w; gw++) {
                    int index = gh * grid_w + gw;
                    if (box[index] < conf_thresh)
                        continue;
                    float16_t max_class_prob = conf[index];
                    int max_class_id = 0;
                    for (int k = 1; k < class_num; k++) {
                        float16_t prob = conf[k * grid_length + index];
                        if (prob > max_class_prob) {
                            max_class_id = k;
                            max_class_prob = prob;
                        }
                    }
                    if (max_class_prob < conf_thresh)
                        continue;
                    int position_tensor_length = output_attrs[position_index].dims[1];
                    vector<float> position_tensor(position_tensor_length);
                    for (int k = 0; k < position_tensor_length; k++)
                        position_tensor[k] = position[index + k * grid_length];
                    vector<float> position = computeDfl(position_tensor);
                    float x0 = (-position[0] + gw + 0.5) * stride;
                    float y0 = (-position[1] + gh + 0.5) * stride;
                    float x1 = (position[2] + gw + 0.5) * stride;
                    float y1 = (position[3] + gh + 0.5) * stride;
                    float class_id = max_class_id;
                    float conf = max_class_prob;
                    result.push_back({x0, y0, x1, y1, class_id, conf});
                }
            }
        }
    }

    nms(result, iou_thresh);
    for (auto& box : result) {
        box[0] = (box[0] - transform_matrix[0]) / transform_matrix[2];
        box[1] = (box[1] - transform_matrix[1]) / transform_matrix[2];
        box[2] = (box[2] - transform_matrix[0]) / transform_matrix[2];
        box[3] = (box[3] - transform_matrix[1]) / transform_matrix[2];
    }
}

bool YOLOV8::inputImage(Mat& image) {
    Mat data = image.clone();
    preprocess(data);
    memcpy(this->input_buffers[0], data.data, data.total() * data.elemSize());
    if (rknn_inputs_set(ctx, io_num.n_input, inputs.data()))
        return false;
    return true;
}

vector<vector<float>> YOLOV8::getResult(float conf_thresh, float iou_thresh) {
    vector<vector<float>> result;
    if (rknn_outputs_get(ctx, io_num.n_output, outputs.data(), NULL))
        return result;
    this->postprocess(result, conf_thresh, iou_thresh);
    return result;
}
