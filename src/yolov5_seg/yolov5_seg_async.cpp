#include <unistd.h>
#include <yolov5_seg/yolov5_seg_async.h>

using namespace std;

pair<vector<vector<float>>, vector<vector<float>>> AsyncYOLOV5SEG::process(Mat data, int worker_id) {
    detectors[worker_id].inputImage(data);
    detectors[worker_id].run();
    return detectors[worker_id].getResult();
}

bool AsyncYOLOV5SEG::init(string path, size_t buffer_length, size_t thread_num) {
    detectors.resize(thread_num);
    for (int i = 0; i < thread_num; i++) {
        switch (i % 3) {
            case 0:
                if (!detectors[i].init(path, RKNN_NPU_CORE_0))
                    return false;
                break;
            case 1:
                if (!detectors[i].init(path, RKNN_NPU_CORE_1))
                    return false;
                break;
            case 2:
                if (!detectors[i].init(path, RKNN_NPU_CORE_2))
                    return false;
                break;
            default:
                break;
        }
    }
    AsyncModule::init(buffer_length, thread_num);
    return true;
}

void AsyncYOLOV5SEG::destroy() {
    AsyncModule::destroy();
    detectors.clear();
    return;
}
