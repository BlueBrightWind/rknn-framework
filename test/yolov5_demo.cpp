#include <chrono>
#include <functional>
#include <opencv2/opencv.hpp>

#include <yolov5/yolov5.h>
#include <yolov5/yolov5_async.h>

using namespace std;
using namespace cv;
using namespace chrono;

static string model_path = "../weights/yolov5s-quantify.rknn";
static string image_path = "../images/bus.jpg";
static int repeat_times = 100;

void yolov5_detector_test() {
    printf("--------------------------------------------\n");
    printf("********** YOLOV5 Detector Test ************\n");
    printf("--------------------------------------------\n");
    YOLOV5 detector;
    detector.init(model_path, RKNN_NPU_CORE_0);
    Mat image = imread(image_path);
    auto t1 = system_clock::now();
    detector.inputImage(image);
    auto t2 = system_clock::now();
    detector.run();
    auto t3 = system_clock::now();
    auto res = detector.getResult();
    auto t4 = system_clock::now();
    printf("Preprocess Cost: %.2f ms\n", duration_cast<microseconds>(t2 - t1).count() / 1000.0);
    printf("Npu Infer Cost: %.2f ms\n", duration_cast<microseconds>(t3 - t2).count() / 1000.0);
    printf("Postprocess Cost: %.2f ms\n", duration_cast<microseconds>(t4 - t3).count() / 1000.0);
    printf("Total Time Cost: %.2f ms\n", duration_cast<microseconds>(t4 - t1).count() / 1000.0);
    for (auto box : res)
        rectangle(image, Point(box[0], box[1]), Point(box[2], box[3]), Scalar(0, 0, 255), 4);
    imwrite("result.jpg", image);
    printf("result save to ./result.jpg\n");
}

void yolov5_detector_speed() {
    printf("--------------------------------------------\n");
    printf("*********** YOLOV5 Speed Test **************\n");
    printf("--------------------------------------------\n");
    YOLOV5 detector;
    detector.init(model_path, RKNN_NPU_CORE_0);
    Mat image = imread(image_path);
    int count = 0;
    int preprocess_cost = 0;
    int infer_cost = 0;
    int postprocess_cost = 0;
    int total_cost = 0;
    while (count++ < repeat_times) {
        auto t1 = system_clock::now();
        detector.inputImage(image);
        auto t2 = system_clock::now();
        detector.run();
        auto t3 = system_clock::now();
        detector.getResult();
        auto t4 = system_clock::now();
        preprocess_cost += duration_cast<microseconds>(t2 - t1).count();
        infer_cost += duration_cast<microseconds>(t3 - t2).count();
        postprocess_cost += duration_cast<microseconds>(t4 - t3).count();
        total_cost += duration_cast<microseconds>(t4 - t1).count();
    }
    printf("Average Time Cost for %d Times:\n", repeat_times);
    printf("Preprocess Cost: %.2f ms\n", preprocess_cost / 1000.0 / repeat_times);
    printf("Npu Infer Cost: %.2f ms\n", infer_cost / 1000.0 / repeat_times);
    printf("Postprocess Cost: %.2f ms\n", postprocess_cost / 1000.0 / repeat_times);
    printf("Total Time Cost: %.2f ms\n", total_cost / 1000.0 / repeat_times);
}

void yolov5_async_test() {
    printf("--------------------------------------------\n");
    printf("******** YOLOV5 Async Speed Test ***********\n");
    printf("--------------------------------------------\n");
    function<void(AsyncYOLOV5&)> putter = [&](AsyncYOLOV5& detector) {
        Mat image = imread(image_path);
        int count = 0;
        while (count++ < repeat_times) {
            detector.put(image);
        }
    };

    function<void(AsyncYOLOV5&)> getter = [&](AsyncYOLOV5& detector) {
        int count = 0;
        auto t1 = system_clock::now();
        while (count++ < repeat_times) {
            auto res = detector.get();
        }
        auto t2 = system_clock::now();
        printf("Async Average Time Cost for %d times: %.2f ms\n", repeat_times, duration_cast<microseconds>(t2 - t1).count() / 1000.0 / repeat_times);
    };

    AsyncYOLOV5 detector;
    detector.init(model_path, 20, 9);
    auto put_thread = thread(putter, ref(detector));
    auto get_thread = thread(getter, ref(detector));
    put_thread.join();
    get_thread.join();
}

int main() {
    yolov5_detector_test();
    yolov5_detector_speed();
    yolov5_async_test();
}