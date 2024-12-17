#include <chrono>
#include <functional>
#include <opencv2/opencv.hpp>

#include <yolov8_seg/yolov8_seg.h>
#include <yolov8_seg/yolov8_seg_async.h>

using namespace std;
using namespace cv;
using namespace chrono;

static string model_path = "../weights/yolov8n-seg.rknn";
static string image_path = "../images/bus.jpg";
static int repeat_times = 100;

void yolov8_seg_detector_test() {
    printf("--------------------------------------------\n");
    printf("******** YOLOV8 Seg Detector Test *********\n");
    printf("--------------------------------------------\n");

    unsigned char class_colors[][3] = {
        {255, 56, 56},    // 'FF3838'
        {255, 157, 151},  // 'FF9D97'
        {255, 112, 31},   // 'FF701F'
        {255, 178, 29},   // 'FFB21D'
        {207, 210, 49},   // 'CFD231'
        {72, 249, 10},    // '48F90A'
        {146, 204, 23},   // '92CC17'
        {61, 219, 134},   // '3DDB86'
        {26, 147, 52},    // '1A9334'
        {0, 212, 187},    // '00D4BB'
        {44, 153, 168},   // '2C99A8'
        {0, 194, 255},    // '00C2FF'
        {52, 69, 147},    // '344593'
        {100, 115, 255},  // '6473FF'
        {0, 24, 236},     // '0018EC'
        {132, 56, 255},   // '8438FF'
        {82, 0, 133},     // '520085'
        {203, 56, 255},   // 'CB38FF'
        {255, 149, 200},  // 'FF95C8'
        {255, 55, 199}    // 'FF37C7'
    };
    int num_colors = 20;

    YOLOV8SEG detector;
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

    // draw mask
    function<int(int, int, int)> clamp = [&](int val, int min, int max) {
        return val > min ? (val < max ? val : max) : min;
    };

    int width = image.cols;
    int height = image.rows;
    float alpha = 0.5f;
    vector<vector<float>>& mask = res.second;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (mask[i][j] == -1)
                continue;
            int pixel_offset = 3 * (i * width + j);
            image.data[pixel_offset + 0] = (unsigned char)clamp(class_colors[(int)mask[i][j] % num_colors][0] * (1 - alpha) + image.data[pixel_offset + 0] * alpha, 0, 255);  // r
            image.data[pixel_offset + 1] = (unsigned char)clamp(class_colors[(int)mask[i][j] % num_colors][1] * (1 - alpha) + image.data[pixel_offset + 1] * alpha, 0, 255);  // r
            image.data[pixel_offset + 2] = (unsigned char)clamp(class_colors[(int)mask[i][j] % num_colors][2] * (1 - alpha) + image.data[pixel_offset + 2] * alpha, 0, 255);  // r
        }
    }

    // draw box
    for (auto box : res.first)
        rectangle(image, Point(box[0], box[1]), Point(box[2], box[3]), Scalar(0, 0, 255), 4);

    imwrite("result.jpg", image);
    printf("result save to ./result.jpg\n");
}

void yolov8_seg_detector_speed() {
    printf("--------------------------------------------\n");
    printf("********* YOLOV8 Seg Speed Test ***********\n");
    printf("--------------------------------------------\n");
    YOLOV8SEG detector;
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

void yolov8_seg_async_test() {
    printf("--------------------------------------------\n");
    printf("****** YOLOV8 Seg Async Speed Test ********\n");
    printf("--------------------------------------------\n");
    function<void(AsyncYOLOV8SEG&)> putter = [&](AsyncYOLOV8SEG& detector) {
        Mat image = imread(image_path);
        int count = 0;
        while (count++ < repeat_times) {
            detector.put(image);
        }
    };

    function<void(AsyncYOLOV8SEG&)> getter = [&](AsyncYOLOV8SEG& detector) {
        int count = 0;
        auto t1 = system_clock::now();
        while (count++ < repeat_times) {
            auto res = detector.get();
        }
        auto t2 = system_clock::now();
        printf("Async Average Time Cost for %d times: %.2f ms\n", repeat_times, duration_cast<microseconds>(t2 - t1).count() / 1000.0 / repeat_times);
    };

    AsyncYOLOV8SEG detector;
    detector.init(model_path, 20, 9);
    auto put_thread = thread(putter, ref(detector));
    auto get_thread = thread(getter, ref(detector));
    put_thread.join();
    get_thread.join();
}

int main() {
    yolov8_seg_detector_test();
    // yolov8_seg_detector_speed();
    // yolov8_seg_async_test();
}