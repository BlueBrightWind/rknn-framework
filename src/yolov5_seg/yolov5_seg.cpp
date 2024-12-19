#include <math.h>
#include <yolov5_seg/yolov5_seg.h>
#include <opencv2/opencv.hpp>

#include <functional>

using namespace std;
using namespace cv;

static int anchors[3][6] = {{10, 13, 16, 30, 33, 23}, {30, 61, 62, 45, 59, 119}, {116, 90, 156, 198, 373, 326}};
static int anchor_num = 3;

bool YOLOV5SEG::init(string path, rknn_core_mask core_mask) {
    if (!BaseModel::init(path, core_mask))
        return false;
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8)
        this->is_quant = true;
    else
        this->is_quant = false;
    return true;
}

void YOLOV5SEG::preprocess(Mat& data) {
    int modelHeight = input_attrs[0].dims[1];
    int modelWidth = input_attrs[0].dims[2];
    int dataHeight = data.rows;
    int dataWidth = data.cols;
    input_size.resize(2);
    input_size[0] = dataWidth;
    input_size[1] = dataHeight;
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

void YOLOV5SEG::postprocess(vector<vector<float>>& boxes, vector<vector<float>>& mask, float conf_thresh, float iou_thresh) {
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

    vector<vector<float>> segments_all;
    vector<vector<float>> segments;

    if (this->is_quant) {
        for (int i = 0; i < io_num.n_output / 2; i++) {
            int box_index = i * 2 + 0;
            int seg_index = i * 2 + 1;
            int8_t* box = (int8_t*)output_buffers[box_index];
            int8_t* seg = (int8_t*)output_buffers[seg_index];
            int box_zp = output_attrs[box_index].zp;
            int seg_zp = output_attrs[seg_index].zp;
            float box_scale = output_attrs[box_index].scale;
            float seg_scale = output_attrs[seg_index].scale;

            int grid_h = output_attrs[box_index].dims[2];
            int grid_w = output_attrs[box_index].dims[3];
            int grid_len = grid_h * grid_w;

            int class_num = output_attrs[box_index].dims[1] / anchor_num - 5;
            int prop_box_size = class_num + 5;
            int segment_size = output_attrs[seg_index].dims[1] / anchor_num;

            int8_t thres_i8 = qntFp32ToAffine(conf_thresh, box_zp, box_scale);
            int stride = sqrt(width * height / grid_w / grid_h);

            for (int gh = 0; gh < grid_h; gh++) {
                for (int gw = 0; gw < grid_w; gw++) {
                    for (int anchor_index = 0; anchor_index < anchor_num; anchor_index++) {
                        int8_t box_confidence = box[(prop_box_size * anchor_index + 4) * grid_len + gh * grid_w + gw];
                        if (box_confidence < thres_i8)
                            continue;
                        int box_offset = (prop_box_size * anchor_index) * grid_len + gh * grid_w + gw;
                        int8_t* in_ptr = box + box_offset;
                        float box_x = deqntAffineToFp32(*in_ptr, box_zp, box_scale) * 2.0 - 0.5;
                        float box_y = deqntAffineToFp32(in_ptr[grid_len], box_zp, box_scale) * 2.0 - 0.5;
                        float box_w = deqntAffineToFp32(in_ptr[2 * grid_len], box_zp, box_scale) * 2.0;
                        float box_h = deqntAffineToFp32(in_ptr[3 * grid_len], box_zp, box_scale) * 2.0;
                        box_x = (box_x + gw) * (float)stride;
                        box_y = (box_y + gh) * (float)stride;
                        box_w = box_w * box_w * (float)anchors[i][anchor_index * 2];
                        box_h = box_h * box_h * (float)anchors[i][anchor_index * 2 + 1];

                        int8_t max_class_prob = in_ptr[5 * grid_len];
                        int max_class_id = 0;
                        for (int k = 1; k < class_num; ++k) {
                            int8_t prob = in_ptr[(5 + k) * grid_len];
                            if (prob > max_class_prob) {
                                max_class_id = k;
                                max_class_prob = prob;
                            }
                        }

                        if (max_class_prob < thres_i8)
                            continue;

                        float x_center = box_x;
                        float y_center = box_y;
                        float w = box_w;
                        float h = box_h;

                        float x0 = x_center - w * 0.5f;
                        float y0 = y_center - h * 0.5f;
                        float x1 = x_center + w * 0.5f;
                        float y1 = y_center + h * 0.5f;

                        x0 = max(min(x0, (float)(width - 1)), 0.f);
                        y0 = max(min(y0, (float)(height - 1)), 0.f);
                        x1 = max(min(x1, (float)(width - 1)), 0.f);
                        y1 = max(min(y1, (float)(height - 1)), 0.f);
                        float conf = deqntAffineToFp32(max_class_prob, box_zp, box_scale);
                        float class_id = max_class_id;
                        float box_index = boxes.size();
                        boxes.push_back({x0, y0, x1, y1, class_id, conf, box_index});

                        vector<float> segment_tensor(segment_size);
                        int seg_offset = (segment_size * anchor_index) * grid_len + gh * grid_w + gw;
                        for (int k = 0; k < segment_size; k++)
                            segment_tensor[k] = deqntAffineToFp32(seg[seg_offset + k * grid_len], seg_zp, seg_scale);
                        segments_all.push_back(segment_tensor);
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < io_num.n_output / 2; i++) {
            int box_index = i * 2 + 0;
            int seg_index = i * 2 + 1;
            float16_t* box = (float16_t*)output_buffers[box_index];
            float16_t* seg = (float16_t*)output_buffers[seg_index];

            int grid_h = output_attrs[box_index].dims[2];
            int grid_w = output_attrs[box_index].dims[3];
            int grid_len = grid_h * grid_w;

            int class_num = output_attrs[box_index].dims[1] / anchor_num - 5;
            int prop_box_size = class_num + 5;
            int segment_size = output_attrs[seg_index].dims[1] / anchor_num;

            int stride = sqrt(width * height / grid_w / grid_h);

            for (int gh = 0; gh < grid_h; gh++) {
                for (int gw = 0; gw < grid_w; gw++) {
                    for (int anchor_index = 0; anchor_index < anchor_num; anchor_index++) {
                        float16_t box_confidence = box[(prop_box_size * anchor_index + 4) * grid_len + gh * grid_w + gw];
                        if (box_confidence < conf_thresh)
                            continue;
                        int box_offset = (prop_box_size * anchor_index) * grid_len + gh * grid_w + gw;
                        float16_t* in_ptr = box + box_offset;
                        float box_x = in_ptr[0] * 2.0 - 0.5;
                        float box_y = in_ptr[grid_len] * 2.0 - 0.5;
                        float box_w = in_ptr[2 * grid_len] * 2.0;
                        float box_h = in_ptr[3 * grid_len] * 2.0;
                        box_x = (box_x + gw) * (float)stride;
                        box_y = (box_y + gh) * (float)stride;
                        box_w = box_w * box_w * (float)anchors[i][anchor_index * 2];
                        box_h = box_h * box_h * (float)anchors[i][anchor_index * 2 + 1];

                        float16_t max_class_prob = in_ptr[5 * grid_len];
                        int max_class_id = 0;
                        for (int k = 1; k < class_num; ++k) {
                            float16_t prob = in_ptr[(5 + k) * grid_len];
                            if (prob > max_class_prob) {
                                max_class_id = k;
                                max_class_prob = prob;
                            }
                        }

                        if (max_class_prob < conf_thresh)
                            continue;

                        float x_center = box_x;
                        float y_center = box_y;
                        float w = box_w;
                        float h = box_h;

                        float x0 = x_center - w * 0.5f;
                        float y0 = y_center - h * 0.5f;
                        float x1 = x_center + w * 0.5f;
                        float y1 = y_center + h * 0.5f;

                        x0 = max(min(x0, (float)(width - 1)), 0.f);
                        y0 = max(min(y0, (float)(height - 1)), 0.f);
                        x1 = max(min(x1, (float)(width - 1)), 0.f);
                        y1 = max(min(y1, (float)(height - 1)), 0.f);
                        float conf = max_class_prob;
                        float class_id = max_class_id;
                        float box_index = boxes.size();
                        boxes.push_back({x0, y0, x1, y1, class_id, conf, box_index});

                        vector<float> segment_tensor(segment_size);
                        int seg_offset = (segment_size * anchor_index) * grid_len + gh * grid_w + gw;
                        for (int k = 0; k < segment_size; k++)
                            segment_tensor[k] = seg[seg_offset + k * grid_len];
                        segments_all.push_back(segment_tensor);
                    }
                }
            }
        }
    }

    // Apply NMS for Boxes and Segments
    nms(boxes, iou_thresh);
    segments.resize(boxes.size());
    for (int i = 0; i < boxes.size(); i++)
        segments[i] = segments_all[boxes[i][6]];

    int proto_height = output_attrs[io_num.n_output - 1].dims[2];
    int proto_width = output_attrs[io_num.n_output - 1].dims[3];
    int proto_channel = output_attrs[io_num.n_output - 1].dims[1];
    vector<vector<float>> proto(proto_channel, vector<float>(proto_height * proto_width));
    if (this->is_quant) {
        int8_t* input = (int8_t*)output_buffers[io_num.n_output - 1];
        int proto_zp = output_attrs[io_num.n_output - 1].zp;
        float proto_scale = output_attrs[io_num.n_output - 1].scale;
        for (int i = 0; i < proto_channel; i++) {
            for (int j = 0; j < proto_height * proto_width; j++) {
                int proto_index = i * proto_height * proto_width + j;
                proto[i][j] = deqntAffineToFp32(input[proto_index], proto_zp, proto_scale);
            }
        }
    } else {
        float16_t* input = (float16_t*)output_buffers[io_num.n_output - 1];
        for (int i = 0; i < proto_channel; i++) {
            for (int j = 0; j < proto_height * proto_width; j++) {
                int proto_index = i * proto_height * proto_width + j;
                proto[i][j] = input[proto_index];
            }
        }
    }

    // Matrix Multiplication
    vector<vector<float>> mat(segments.size(), vector<float>(proto_height * proto_width));
    for (int i = 0; i < segments.size(); i++) {
        for (int j = 0; j < proto_height * proto_width; j++) {
            float val = 0;
            for (int k = 0; k < proto_channel; k++)
                val += segments[i][k] * proto[k][j];
            mat[i][j] = val;
        }
    }

    // Resize Matrix
    vector<Mat> mats(segments.size());
    for (int i = 0; i < segments.size(); i++) {
        Mat data(proto_height, proto_width, CV_32F, mat[i].data());
        resize(data, mats[i], Size(width, height), 0, 0, INTER_LINEAR);
    }

    // Get Origin Mask
    Mat origin_mask(height, width, CV_32F, -1.0f);
    for (int i = 0; i < segments.size(); i++) {
        float x1 = boxes[i][0];
        float y1 = boxes[i][1];
        float x2 = boxes[i][2];
        float y2 = boxes[i][3];
        float class_id = boxes[i][4];

        Rect roi(Point(x1, y1), Point(x2, y2));
        Mat origin_mask_roi = origin_mask(roi);
        Mat mat_roi = mats[i](roi);

        Mat mask_mat_set = (mat_roi > 0);
        Mat mask_mat_copy = (origin_mask_roi < 0);
        origin_mask_roi.setTo(class_id, mask_mat_set & mask_mat_copy);
    }

    // Crop Mask
    int crop_width = input_size[0] * transform_matrix[2];
    int crop_height = input_size[1] * transform_matrix[2];
    int x_min = transform_matrix[0];
    int y_min = transform_matrix[1];
    int x_max = x_min + crop_width;
    int y_max = y_min + crop_height;
    Mat crop_mask = origin_mask(Rect(Point(x_min, y_min), Point(x_max, y_max)));

    // Reverse Mask
    Mat image_mask;
    resize(crop_mask, image_mask, Size(input_size[0], input_size[1]), 0, 0, INTER_LINEAR);
    vector<float> mask_data = image_mask.reshape(1, 1);
    mask.resize(input_size[1]);
    for (int i = 0; i < mask.size(); i++) {
        mask[i].resize(input_size[0]);
        int offset = i * input_size[0];
        float* row_ptr = mask_data.data() + offset;
        memcpy(mask[i].data(), row_ptr, input_size[0] * sizeof(float));
    }

    for (auto& box : boxes) {
        box[0] = (box[0] - transform_matrix[0]) / transform_matrix[2];
        box[1] = (box[1] - transform_matrix[1]) / transform_matrix[2];
        box[2] = (box[2] - transform_matrix[0]) / transform_matrix[2];
        box[3] = (box[3] - transform_matrix[1]) / transform_matrix[2];
        box.pop_back();
    }
}

bool YOLOV5SEG::inputImage(Mat& image) {
    Mat data = image.clone();
    preprocess(data);
    memcpy(this->input_buffers[0], data.data, data.total() * data.elemSize());
    if (rknn_inputs_set(ctx, io_num.n_input, inputs.data()))
        return false;
    return true;
}

pair<vector<vector<float>>, vector<vector<float>>> YOLOV5SEG::getResult(float conf_thresh, float iou_thresh) {
    vector<vector<float>> boxes;
    vector<vector<float>> mask;
    if (rknn_outputs_get(ctx, io_num.n_output, outputs.data(), NULL))
        return {boxes, mask};
    this->postprocess(boxes, mask, conf_thresh, iou_thresh);
    return {boxes, mask};
}
