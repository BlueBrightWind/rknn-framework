#include <math.h>
#include <yolov8_pose/yolov8_pose.h>

#include <functional>

using namespace std;

static int dfl_length = 16;

bool YOLOV8POSE::init(string path, rknn_core_mask core_mask) {
    if (!BaseModel::init(path, core_mask))
        return false;
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8)
        this->is_quant = true;
    else
        this->is_quant = false;
    return true;
}

void YOLOV8POSE::preprocess(Mat& data) {
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

void YOLOV8POSE::postprocess(vector<vector<float>>& boxes, vector<vector<vector<float>>>& keypoints, float conf_thresh, float iou_thresh) {
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

    function<float(float)> sigmoid = [&](float x) {
        return 1.0 / (1.0 + expf(-x));
    };

    function<float(float)> unsigmoid = [&](float y) {
        return -1.0 * logf((1.0 / y) - 1.0);
    };

    function<vector<float>(vector<float>)> softmax = [&](vector<float> tensor) {
        vector<float> output = vector<float>(tensor.size());
        float max_val = *max_element(tensor.begin(), tensor.end());
        float sum_exp = 0.0;
        for (int i = 0; i < tensor.size(); i++)
            sum_exp += expf(tensor[i] - max_val);
        for (int i = 0; i < tensor.size(); ++i)
            output[i] = expf(tensor[i] - max_val) / sum_exp;
        return output;
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
    int position_length = dfl_length * 4;

    if (this->is_quant) {
        int base_keypoint_index = 0;
        for (int i = 0; i < io_num.n_output - 1; i++) {
            int8_t* input = (int8_t*)output_buffers[i];
            int class_num = output_attrs[i].dims[1] - position_length;
            int zp = output_attrs[i].zp;
            float scale = output_attrs[i].scale;

            int grid_h = output_attrs[i].dims[2];
            int grid_w = output_attrs[i].dims[3];
            int grid_length = grid_h * grid_w;
            int stride = sqrt(width * height / grid_w / grid_h);

            int8_t thres_i8 = qntFp32ToAffine(unsigmoid(conf_thresh), zp, scale);

            for (int gh = 0; gh < grid_h; gh++) {
                for (int gw = 0; gw < grid_w; gw++) {
                    for (int c = 0; c < class_num; c++) {
                        int class_index = (position_length + c) * grid_length + gh * grid_w + gw;
                        if (input[class_index] < thres_i8)
                            continue;
                        vector<vector<float>> position(position_length / dfl_length);
                        for (int j = 0; j < position.size(); j++) {
                            vector<float> position_segment(dfl_length);
                            for (int k = 0; k < dfl_length; k++) {
                                int index = j * dfl_length + k;
                                int position_index = index * grid_length + gh * grid_w + gw;
                                position_segment[k] = deqntAffineToFp32(input[position_index], zp, scale);
                            }
                            position[j] = position_segment;
                        }

                        for (int j = 0; j < position.size(); j++)
                            position[j] = softmax(position[j]);

                        float x1 = 0.0;
                        float y1 = 0.0;
                        float x2 = 0.0;
                        float y2 = 0.0;

                        for (int j = 0; j < dfl_length; j++) {
                            x1 += position[0][j] * j;
                            y1 += position[1][j] * j;
                            x2 += position[2][j] * j;
                            y2 += position[3][j] * j;
                        }

                        x1 = (gw + 0.5 - x1) * stride;
                        y1 = (gh + 0.5 - y1) * stride;
                        x2 = (gw + 0.5 + x2) * stride;
                        y2 = (gh + 0.5 + y2) * stride;
                        float class_id = c;
                        float conf = sigmoid(deqntAffineToFp32(input[class_index], zp, scale));
                        float keypoint_index = base_keypoint_index + gh * grid_w + gw;
                        boxes.push_back({x1, y1, x2, y2, class_id, conf, keypoint_index});
                    }
                }
            }
            base_keypoint_index += grid_length;
        }
    } else {
        int base_keypoint_index = 0;
        for (int i = 0; i < io_num.n_output - 1; i++) {
            float16_t* input = (float16_t*)output_buffers[i];
            int class_num = output_attrs[i].dims[1] - position_length;

            int grid_h = output_attrs[i].dims[2];
            int grid_w = output_attrs[i].dims[3];
            int grid_length = grid_h * grid_w;
            int stride = sqrt(width * height / grid_w / grid_h);

            float thres_fp32 = unsigmoid(conf_thresh);

            for (int gh = 0; gh < grid_h; gh++) {
                for (int gw = 0; gw < grid_w; gw++) {
                    for (int c = 0; c < class_num; c++) {
                        int class_index = (position_length + c) * grid_length + gh * grid_w + gw;
                        if (input[class_index] < thres_fp32)
                            continue;
                        vector<vector<float>> position(position_length / dfl_length);
                        for (int j = 0; j < position.size(); j++) {
                            vector<float> position_segment(dfl_length);
                            for (int k = 0; k < dfl_length; k++) {
                                int index = j * dfl_length + k;
                                int position_index = index * grid_length + gh * grid_w + gw;
                                position_segment[k] = input[position_index];
                            }
                            position[j] = position_segment;
                        }

                        for (int j = 0; j < position.size(); j++)
                            position[j] = softmax(position[j]);

                        float x1 = 0.0;
                        float y1 = 0.0;
                        float x2 = 0.0;
                        float y2 = 0.0;

                        for (int j = 0; j < dfl_length; j++) {
                            x1 += position[0][j] * j;
                            y1 += position[1][j] * j;
                            x2 += position[2][j] * j;
                            y2 += position[3][j] * j;
                        }

                        x1 = (gw + 0.5 - x1) * stride;
                        y1 = (gh + 0.5 - y1) * stride;
                        x2 = (gw + 0.5 + x2) * stride;
                        y2 = (gh + 0.5 + y2) * stride;
                        float class_id = c;
                        float conf = sigmoid(input[class_index]);
                        float keypoint_index = base_keypoint_index + gh * grid_w + gw;
                        boxes.push_back({x1, y1, x2, y2, class_id, conf, keypoint_index});
                    }
                }
            }
            base_keypoint_index += grid_length;
        }
    }

    nms(boxes, iou_thresh);

    if (this->is_quant) {
        int8_t* input = (int8_t*)output_buffers[io_num.n_output - 1];
        int keypoint_num = output_attrs[io_num.n_output - 1].dims[1];
        int keypoint_info_length = output_attrs[io_num.n_output - 1].dims[2];
        int keypoint_length = output_attrs[io_num.n_output - 1].dims[3];

        int zp = output_attrs[io_num.n_output - 1].zp;
        float scale = output_attrs[io_num.n_output - 1].scale;

        for (int i = 0; i < boxes.size(); i++) {
            vector<vector<float>> boxes_keypoints;
            for (int j = 0; j < keypoint_num; j++) {
                int keypoint_index = boxes[i][6];
                vector<float> keypoint(keypoint_info_length);
                for (int k = 0; k < keypoint_info_length; k++)
                    keypoint[k] = deqntAffineToFp32(input[j * keypoint_info_length * keypoint_length + k * keypoint_length + keypoint_index], zp, scale);
                boxes_keypoints.push_back(keypoint);
            }
            keypoints.push_back(boxes_keypoints);
        }
    } else {
        float16_t* input = (float16_t*)output_buffers[io_num.n_output - 1];
        int keypoint_num = output_attrs[io_num.n_output - 1].dims[1];
        int keypoint_info_length = output_attrs[io_num.n_output - 1].dims[2];
        int keypoint_length = output_attrs[io_num.n_output - 1].dims[3];

        for (int i = 0; i < boxes.size(); i++) {
            vector<vector<float>> boxes_keypoints;
            for (int j = 0; j < keypoint_num; j++) {
                int keypoint_index = boxes[i][6];
                vector<float> keypoint(keypoint_info_length);
                for (int k = 0; k < keypoint_info_length; k++)
                    keypoint[k] = input[j * keypoint_info_length * keypoint_length + k * keypoint_length + keypoint_index];
                boxes_keypoints.push_back(keypoint);
            }
            keypoints.push_back(boxes_keypoints);
        }
    }

    for (auto& box : boxes) {
        box[0] = (box[0] - transform_matrix[0]) / transform_matrix[2];
        box[1] = (box[1] - transform_matrix[1]) / transform_matrix[2];
        box[2] = (box[2] - transform_matrix[0]) / transform_matrix[2];
        box[3] = (box[3] - transform_matrix[1]) / transform_matrix[2];
        box.pop_back();
    }

    for (int i = 0; i < keypoints.size(); i++) {
        for (auto& keypoint : keypoints[i]) {
            keypoint[0] = (keypoint[0] - transform_matrix[0]) / transform_matrix[2];
            keypoint[1] = (keypoint[1] - transform_matrix[1]) / transform_matrix[2];
        }
    }
}

bool YOLOV8POSE::inputImage(Mat& image) {
    Mat data = image.clone();
    preprocess(data);
    memcpy(this->input_buffers[0], data.data, data.total() * data.elemSize());
    if (rknn_inputs_set(ctx, io_num.n_input, inputs.data()))
        return false;
    return true;
}

pair<vector<vector<float>>, vector<vector<vector<float>>>> YOLOV8POSE::getResult(float conf_thresh, float iou_thresh) {
    vector<vector<float>> boxes;
    vector<vector<vector<float>>> keypoints;
    if (rknn_outputs_get(ctx, io_num.n_output, outputs.data(), NULL))
        return {boxes, keypoints};
    this->postprocess(boxes, keypoints, conf_thresh, iou_thresh);
    return {boxes, keypoints};
}
