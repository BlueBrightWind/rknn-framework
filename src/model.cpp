#include <model.h>

#include <cstring>
#include <fstream>
#include <iostream>

using namespace std;

BaseModel::BaseModel() {
}

BaseModel::~BaseModel() {
    this->destroy();
}

bool BaseModel::init(string path, rknn_core_mask core_mask) {
    // Load Model
    ifstream file(path, ios::binary);
    if (!file.is_open())
        return false;
    file.seekg(0, ios::end);
    int model_size = file.tellg();
    file.seekg(0, ios::beg);
    this->model = new char[model_size];
    file.read(model, model_size);
    file.close();

    // Init Context
    if (rknn_init(&ctx, model, model_size, 0, NULL)) {
        return false;
    }

    // Set Core Mask
    if (rknn_set_core_mask(ctx, core_mask)) {
        return false;
    }

    // Set Attr Num
    if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(rknn_input_output_num))) {
        return false;
    }

    // Set Input Attr
    this->input_attrs.resize(io_num.n_input);
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        if (rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr)))
            return false;
    }

    // Set Output Attr
    this->output_attrs.resize(io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        if (rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr)))
            return false;
    }

    // Init Model Inputs
    this->inputs.resize(io_num.n_input);
    this->input_buffers.resize(io_num.n_input);
    for (int i = 0; i < io_num.n_input; i++) {
        memset(&inputs[i], 0, sizeof(rknn_input));
    }
    this->initModelInputs();

    // Init Model Outputs
    this->outputs.resize(io_num.n_output);
    this->output_buffers.resize(io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++) {
        memset(&outputs[i], 0, sizeof(rknn_output));
    }
    this->initModelOutputs();

    return true;
}

bool BaseModel::destroy() {
    // Release Model Input and Output
    this->releaseModelInputs();
    this->releaseModelOutputs();

    // Release RKNN Context
    if (rknn_destroy(ctx)) {
        return false;
    }

    // Release Model Data
    if (model) {
        delete[] model;
    }

    return true;
}

bool BaseModel::run() {
    if (rknn_run(ctx, NULL))
        return false;
    return true;
}

bool BaseModel::initModelInputs() {
    for (int i = 0; i < io_num.n_input; i++) {
        inputs[i].index = i;
        inputs[i].pass_through = false;
        inputs[i].type = RKNN_TENSOR_UINT8;
        inputs[i].fmt = RKNN_TENSOR_NHWC;
        inputs[i].size = input_attrs[i].size;
        input_buffers[i] = new char[input_attrs[i].size];
        inputs[i].buf = input_buffers[i];
    }
    return true;
}

bool BaseModel::initModelOutputs() {
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].want_float = false;
        outputs[i].is_prealloc = true;
        outputs[i].index = i;
        outputs[i].size = output_attrs[i].size;
        output_buffers[i] = new char[output_attrs[i].size];
        outputs[i].buf = output_buffers[i];
    }
    return true;
}

bool BaseModel::releaseModelInputs() {
    for (int i = 0; i < io_num.n_input; i++) {
        delete[] (char*)(input_buffers[i]);
    }
    return true;
}

bool BaseModel::releaseModelOutputs() {
    for (int i = 0; i < io_num.n_output; i++) {
        delete[] (char*)(output_buffers[i]);
    }
    return true;
}
