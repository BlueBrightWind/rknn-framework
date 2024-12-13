#include <yolov5.h>
#include <async.hpp>

using namespace std;

class AsyncYOLOV5 : public AsyncModule<Mat, vector<vector<float>>> {
   private:
    vector<YOLOV5> detectors;

   private:
    vector<vector<float>> process(Mat data, int worker_id);

   public:
    bool init(string path, size_t buffer_length, size_t thread_num);
    void destroy();
};
