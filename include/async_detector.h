#include <detector.h>
#include <async.hpp>

using namespace std;

class AsyncDetector : public AsyncModule<Mat, vector<vector<float>>> {
   private:
    vector<Detector> detectors;

   private:
    vector<vector<float>> process(Mat data, int worker_id);

   public:
    bool init(string path, size_t buffer_length, size_t thread_num);
    void destroy();
};
