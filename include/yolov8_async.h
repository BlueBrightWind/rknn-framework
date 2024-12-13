#include <yolov8.h>
#include <async.hpp>

using namespace std;

class AsyncYOLOV8 : public AsyncModule<Mat, vector<vector<float>>> {
   private:
    vector<YOLOV8> detectors;

   private:
    vector<vector<float>> process(Mat data, int worker_id);

   public:
    bool init(string path, size_t buffer_length, size_t thread_num);
    void destroy();
};
