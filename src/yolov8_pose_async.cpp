#include <unistd.h>
#include <yolov8_pose_async.h>

using namespace std;

pair<vector<vector<float>>, vector<vector<vector<float>>>> AsyncYOLOV8POSE::process(Mat data, int worker_id) {
    detectors[worker_id].inputImage(data);
    detectors[worker_id].run();
    return detectors[worker_id].getResult();
}

bool AsyncYOLOV8POSE::init(string path, size_t buffer_length, size_t thread_num) {
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

void AsyncYOLOV8POSE::destroy() {
    AsyncModule::destroy();
    detectors.clear();
    return;
}
