#include <unistd.h>
#include <yolov5.h>
#include <yolov5_async.h>
#include <yolov8.h>
#include <yolov8_async.h>

#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void yolov8_detector_test() {
    YOLOV8 detector;
    detector.init("/root/rknn_framework/weights/yolov8n.rknn", RKNN_NPU_CORE_0);
    Mat image = imread("/root/rknn_framework/bus.jpg");
    auto t1 = chrono::system_clock::now();
    detector.inputImage(image);
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
    for (auto box : res)
        rectangle(image, Point(box[0], box[1]), Point(box[2], box[3]), Scalar(0, 0, 255), 4);
    imwrite("result.jpg", image);
}

void yolov8_detector_speed() {
    YOLOV8 detector;
    detector.init("/root/rknn_framework/weights/yolov8n.rknn", RKNN_NPU_CORE_0);
    Mat image = imread("/root/rknn_framework/test1.png");
    while (1) {
        auto t1 = chrono::system_clock::now();
        detector.inputImage(image);
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
}

void yolov8_async_test() {
    function<void(AsyncYOLOV8&)> putter = [&](AsyncYOLOV8& detector) {
        Mat image = imread("/root/rknn_framework/test1.png");
        while (1) {
            detector.put(image);
            // usleep(5000);
            detector.put(image);
            // usleep(5000);
        }
    };

    function<void(AsyncYOLOV8&)> getter = [&](AsyncYOLOV8& detector) {
        while (1) {
            auto t1 = chrono::system_clock::now();
            auto res = detector.get();
            usleep(6000);
            auto t2 = chrono::system_clock::now();
            printf("get result time cost: %.2f ms\n", chrono::duration_cast<chrono::microseconds>(t2 - t1).count() / 1000.0);
        }
    };

    AsyncYOLOV8 detector;
    detector.init("/root/rknn_framework/weights/yolov8n-quantify.rknn", 20, 9);
    auto put_thread = thread(putter, ref(detector));
    auto get_thread = thread(getter, ref(detector));
    put_thread.join();
    get_thread.join();
}

void yolov5_detector_test() {
    YOLOV5 detector;
    detector.init("/root/rknn_framework/weights/yolov5s-quantify.rknn", RKNN_NPU_CORE_0);
    Mat image = imread("/root/rknn_framework/test1.png");
    auto t1 = chrono::system_clock::now();
    detector.inputImage(image);
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
    for (auto box : res)
        rectangle(image, Point(box[0], box[1]), Point(box[2], box[3]), Scalar(0, 0, 255), 4);
    imwrite("result.jpg", image);
}

void yolov5_detector_speed() {
    YOLOV5 detector;
    detector.init("/root/rknn_framework/weights/yolov5s-quantify.rknn", RKNN_NPU_CORE_0_1_2);
    Mat image = imread("/root/rknn_framework/test1.png");
    while (1) {
        auto t1 = chrono::system_clock::now();
        detector.inputImage(image);
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
}

void yolov5_async_test() {
    function<void(AsyncYOLOV5&)> putter = [&](AsyncYOLOV5& detector) {
        Mat image = imread("/root/rknn_framework/test1.png");
        while (1) {
            detector.put(image);
            // usleep(5000);
            detector.put(image);
            // usleep(5000);
        }
    };

    function<void(AsyncYOLOV5&)> getter = [&](AsyncYOLOV5& detector) {
        while (1) {
            auto t1 = chrono::system_clock::now();
            auto res = detector.get();
            usleep(14000);
            auto t2 = chrono::system_clock::now();
            printf("get result time cost: %.2f ms\n", chrono::duration_cast<chrono::microseconds>(t2 - t1).count() / 1000.0);
        }
    };

    AsyncYOLOV5 detector;
    detector.init("/root/rknn_framework/weights/yolov5s-quantify.rknn", 20, 9);
    auto put_thread = thread(putter, ref(detector));
    auto get_thread = thread(getter, ref(detector));
    put_thread.join();
    get_thread.join();
}

int main() {
    // yolov5_detector_speed();
    // yolov8_detector_speed();
    // yolov8_detector_test();
    yolov8_async_test();
}