#include <yolov8_pose/yolov8_pose.h>
#include <base/async.hpp>

using namespace std;

class AsyncYOLOV8POSE : public AsyncModule<Mat, pair<vector<vector<float>>, vector<vector<vector<float>>>>> {
   private:
    vector<YOLOV8POSE> detectors;

   private:
    pair<vector<vector<float>>, vector<vector<vector<float>>>> process(Mat data, int worker_id);

   public:
    bool init(string path, size_t buffer_length, size_t thread_num);
    void destroy();
};
