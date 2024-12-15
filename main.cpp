#include <unistd.h>
#include <yolo11/yolo11.h>
#include <yolo11/yolo11_async.h>
#include <yolov5/yolov5.h>
#include <yolov5/yolov5_async.h>
#include <yolov7/yolov7.h>
#include <yolov7/yolov7_async.h>
#include <yolov8/yolov8.h>
#include <yolov8/yolov8_async.h>
#include <yolov8_pose/yolov8_pose.h>
#include <yolov8_pose/yolov8_pose_async.h>

#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void yolov8_pose_test() {
    int skeleton[38] = {16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8,
                        7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7};
    YOLOV8POSE detector;
    detector.init("/root/rknn_framework/weights/yolov8n-pose.rknn", RKNN_NPU_CORE_0);
    Mat image = imread("/root/rknn_framework/pose.jpg");
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
    for (auto box : res.first)
        rectangle(image, Point(box[0], box[1]), Point(box[2], box[3]), Scalar(0, 0, 255), 4);

    for (auto keypoints : res.second) {
        for (int i = 0; i < 38 / 2; i++)
            line(image, Point(keypoints[skeleton[2 * i] - 1][0], keypoints[skeleton[2 * i] - 1][1]), Point(keypoints[skeleton[2 * i + 1] - 1][0], keypoints[skeleton[2 * i + 1] - 1][1]), Scalar(0, 255, 0), 2);

        for (int i = 0; i < 17; i++)
            circle(image, Point(keypoints[i][0], keypoints[i][1]), 4, Scalar(0, 255, 0), -1);
    }
    imwrite("result.jpg", image);
}

void yolov8_pose_speed() {
    int skeleton[38] = {16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8,
                        7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7};
    YOLOV8POSE detector;
    detector.init("/root/rknn_framework/weights/yolov8n-pose.rknn", RKNN_NPU_CORE_0);
    Mat image = imread("/root/rknn_framework/pose2.jpg");
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

void yolov8_pose_async_test() {
    function<void(AsyncYOLOV8POSE&)> putter = [&](AsyncYOLOV8POSE& detector) {
        Mat image = imread("/root/rknn_framework/pose.jpg");
        while (1) {
            detector.put(image);
            // usleep(5000);
            detector.put(image);
            // usleep(5000);
        }
    };

    function<void(AsyncYOLOV8POSE&)> getter = [&](AsyncYOLOV8POSE& detector) {
        while (1) {
            auto t1 = chrono::system_clock::now();
            auto res = detector.get();
            usleep(15000);
            auto t2 = chrono::system_clock::now();
            printf("get result time cost: %.2f ms\n", chrono::duration_cast<chrono::microseconds>(t2 - t1).count() / 1000.0);
        }
    };

    AsyncYOLOV8POSE detector;
    detector.init("/root/rknn_framework/weights/yolov8n-pose.rknn", 20, 9);
    auto put_thread = thread(putter, ref(detector));
    auto get_thread = thread(getter, ref(detector));
    put_thread.join();
    get_thread.join();
}

void yolo11_detector_test() {
    YOLOV8 detector;
    detector.init("/root/rknn_framework/weights/yolo11n-quantify.rknn", RKNN_NPU_CORE_0);
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

void yolo11_detector_speed() {
    YOLOV8 detector;
    detector.init("/root/rknn_framework/weights/yolo11n-quantify.rknn", RKNN_NPU_CORE_0_1_2);
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

void yolo11_async_test() {
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
            usleep(8000);
            auto t2 = chrono::system_clock::now();
            printf("get result time cost: %.2f ms\n", chrono::duration_cast<chrono::microseconds>(t2 - t1).count() / 1000.0);
        }
    };

    AsyncYOLOV8 detector;
    detector.init("/root/rknn_framework/weights/yolo11n-quantify.rknn", 20, 9);
    auto put_thread = thread(putter, ref(detector));
    auto get_thread = thread(getter, ref(detector));
    put_thread.join();
    get_thread.join();
}

void yolov8_detector_test() {
    YOLOV8 detector;
    detector.init("/root/rknn_framework/weights/yolov8n-quantify.rknn", RKNN_NPU_CORE_0);
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
    detector.init("/root/rknn_framework/weights/yolov8n-quantify.rknn", RKNN_NPU_CORE_0_1_2);
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
            usleep(8000);
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

void yolov7_detector_test() {
    YOLOV7 detector;
    detector.init("/root/rknn_framework/weights/yolov7-tiny-quantify.rknn", RKNN_NPU_CORE_0);
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

void yolov7_detector_speed() {
    YOLOV5 detector;
    detector.init("/root/rknn_framework/weights/yolov7-tiny-quantify.rknn", RKNN_NPU_CORE_0_1_2);
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

void yolov7_async_test() {
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
    detector.init("/root/rknn_framework/weights/yolov7-tiny-quantify.rknn", 20, 9);
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
    // yolo11_detector_test();
    // yolo11_detector_speed();
    // yolo11_async_test();

    // yolov8_detector_test();
    // yolov8_detector_speed();
    // yolov8_async_test();

    // yolov7_detector_test();
    // yolov7_detector_speed();
    // yolov7_async_test();

    // yolov5_detector_speed();
    // yolov8_detector_speed();
    // yolov8_detector_test();

    // yolov8_pose_test();
    // yolov8_pose_speed();
    yolov8_pose_async_test();
}