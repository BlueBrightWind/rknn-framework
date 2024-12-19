#include <yolov5_seg/yolov5_seg.h>
#include <base/async.hpp>

using namespace std;

class AsyncYOLOV5SEG : public AsyncModule<Mat, pair<vector<vector<float>>, vector<vector<float>>>> {
   private:
    vector<YOLOV5SEG> detectors;

   private:
    pair<vector<vector<float>>, vector<vector<float>>> process(Mat data, int worker_id);

   public:
    bool init(string path, size_t buffer_length, size_t thread_num);
    void destroy();
};
