#include <chrono>
#include <functional>
#include <opencv2/opencv.hpp>

#include <yolov8_pose/yolov8_pose.h>
#include <yolov8_pose/yolov8_pose_async.h>

using namespace std;
using namespace cv;
using namespace chrono;

static string model_path = "../weights/yolov8n-pose.rknn";
static string image_path = "../images/pose.jpg";
static int repeat_times = 100;

void yolov8_pose_detector_test() {
    printf("--------------------------------------------\n");
    printf("******** YOLOV8 Pose Detector Test *********\n");
    printf("--------------------------------------------\n");
    int skeleton[38] = {16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8,
                        7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7};
    YOLOV8POSE detector;
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

    for (auto box : res.first)
        rectangle(image, Point(box[0], box[1]), Point(box[2], box[3]), Scalar(0, 0, 255), 4);

    for (auto keypoints : res.second) {
        for (int i = 0; i < 38 / 2; i++)
            line(image, Point(keypoints[skeleton[2 * i] - 1][0], keypoints[skeleton[2 * i] - 1][1]), Point(keypoints[skeleton[2 * i + 1] - 1][0], keypoints[skeleton[2 * i + 1] - 1][1]), Scalar(0, 255, 0), 2);
        for (int i = 0; i < 17; i++)
            circle(image, Point(keypoints[i][0], keypoints[i][1]), 4, Scalar(0, 255, 0), -1);
    }

    imwrite("result.jpg", image);
    printf("result save to ./result.jpg\n");
}

void yolov8_pose_detector_speed() {
    printf("--------------------------------------------\n");
    printf("********* YOLOV8 Pose Speed Test ***********\n");
    printf("--------------------------------------------\n");
    YOLOV8POSE detector;
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

void yolov8_pose_async_test() {
    printf("--------------------------------------------\n");
    printf("****** YOLOV8 Pose Async Speed Test ********\n");
    printf("--------------------------------------------\n");
    function<void(AsyncYOLOV8POSE&)> putter = [&](AsyncYOLOV8POSE& detector) {
        Mat image = imread(image_path);
        int count = 0;
        while (count++ < repeat_times) {
            detector.put(image);
        }
    };

    function<void(AsyncYOLOV8POSE&)> getter = [&](AsyncYOLOV8POSE& detector) {
        int count = 0;
        auto t1 = system_clock::now();
        while (count++ < repeat_times) {
            auto res = detector.get();
        }
        auto t2 = system_clock::now();
        printf("Async Average Time Cost for %d times: %.2f ms\n", repeat_times, duration_cast<microseconds>(t2 - t1).count() / 1000.0 / repeat_times);
    };

    AsyncYOLOV8POSE detector;
    detector.init(model_path, 20, 9);
    auto put_thread = thread(putter, ref(detector));
    auto get_thread = thread(getter, ref(detector));
    put_thread.join();
    get_thread.join();
}

int main() {
    yolov8_pose_detector_test();
    yolov8_pose_detector_speed();
    yolov8_pose_async_test();
}