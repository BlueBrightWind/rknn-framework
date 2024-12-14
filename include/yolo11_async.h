#include <yolo11.h>
#include <async.hpp>

using namespace std;

class AsyncYOLO11 : public AsyncModule<Mat, vector<vector<float>>> {
   private:
    vector<YOLO11> detectors;

   private:
    vector<vector<float>> process(Mat data, int worker_id);

   public:
    bool init(string path, size_t buffer_length, size_t thread_num);
    void destroy();
};
