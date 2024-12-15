#include <yolov7/yolov7.h>
#include <base/async.hpp>

using namespace std;

class AsyncYOLOV7 : public AsyncModule<Mat, vector<vector<float>>> {
   private:
    vector<YOLOV7> detectors;

   private:
    vector<vector<float>> process(Mat data, int worker_id);

   public:
    bool init(string path, size_t buffer_length, size_t thread_num);
    void destroy();
};
