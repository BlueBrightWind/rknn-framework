#include <async_detector.h>
#include <detector.h>

#include <unistd.h>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void putter(AsyncDetector& detector) {
    Mat image;
    while (1) {
        image = imread("/root/rknn_framework/test1.png");
        detector.put(image);
        // usleep(5000);
        image  = imread("/root/rknn_framework/test1.png");
        detector.put(image);
        // usleep(5000);
    }
}

void getter(AsyncDetector& detector) {
    while (1) {
        auto t1 = chrono::system_clock::now();
        auto res = detector.get();
        usleep(14000);
        auto t2 = chrono::system_clock::now();
        printf("get result time cost: %.2f ms\n", chrono::duration_cast<chrono::microseconds>(t2 - t1).count() / 1000.0);
    }
}

int main() {
    Detector detector;
    detector.init("/root/rknn_framework/yolov7-tiny-1280-quantify.rknn", RKNN_NPU_CORE_0_1_2);
    Mat image1 = imread("/root/rknn_framework/test1.png");
    Mat image2 = imread("/root/rknn_framework/test2.jpg");

    while (1) {
        auto t1 = chrono::system_clock::now();
        detector.inputImage(image1);
        auto t2 = chrono::system_clock::now();
        detector.run();
        auto t3 = chrono::system_clock::now();
        auto res = detector.getResult();
        auto t4 = chrono::system_clock::now();
        printf("preprocess cost: %.2f, inter cost: %.2f, postprocess cost: %.2f, time cost: %.2f ms\n",
               chrono::duration_cast<chrono::microseconds>(t2 - t1).count() / 1000.0,
               chrono::duration_cast<chrono::microseconds>(t3 - t2).count() / 1000.0,
               chrono::duration_cast<chrono::microseconds>(t4 - t3).count() / 1000.0,
               chrono::duration_cast<chrono::microseconds>(t4 - t1).count() / 1000.0);
    }

    // AsyncDetector detector;
    // detector.init("/root/rknn_framework/yolov7-tiny-1280-quantify.rknn", 20, 9);
    // auto put_thread = thread(putter, ref(detector));
    // auto get_thread = thread(getter, ref(detector));
    // put_thread.join();
    // get_thread.join();
}