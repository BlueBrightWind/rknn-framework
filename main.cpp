#include <yolov8_seg/yolov8_seg_async.h>

#include <chrono>
#include <functional>
#include <opencv2/opencv.hpp>
#include <queue>

using namespace std;
using namespace cv;
using namespace chrono;

static string model_path = "../weights/yolov8n-seg.rknn";
static string video_path = "../videos/people.mp4";

int main() {
    AsyncYOLOV8SEG detector;
    detector.init(model_path, 20, 9);

    queue<Mat> frames;
    VideoCapture cap("filesrc location=" + video_path + " ! qtdemux ! h264parse ! mppvideodec width=1920 height=1080 format=16 ! appsink sync=false");
    VideoWriter writer("appsrc ! mpph264enc ! h264parse ! mp4mux ! filesink location=./result.mp4 sync=false", 0, 30.0, Size(1920, 1080));

    function<int(int, int, int)> clamp = [&](int val, int min, int max) {
        return val > min ? (val < max ? val : max) : min;
    };

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

    for (int i = 0; i < 15; i++) {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
        Mat data = frame;
        detector.put(data);
        frames.push(frame);
    }

    int count = 0;
    while (1) {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
        Mat data = frame;
        detector.put(data);
        frames.push(frame);

        auto res = detector.get();
        Mat c_frame = frames.front();
        frames.pop();

        int width = c_frame.cols;
        int height = c_frame.rows;
        float alpha = 0.5f;
        Mat& mask = res.second;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (mask.at<float>(i, j) == -1)
                    continue;
                int pixel_offset = 3 * (i * width + j);
                c_frame.data[pixel_offset + 0] = (unsigned char)clamp(class_colors[(int)mask.at<float>(i, j) % num_colors][0] * (1 - alpha) + c_frame.data[pixel_offset + 0] * alpha, 0, 255);  // r
                c_frame.data[pixel_offset + 1] = (unsigned char)clamp(class_colors[(int)mask.at<float>(i, j) % num_colors][1] * (1 - alpha) + c_frame.data[pixel_offset + 1] * alpha, 0, 255);  // g
                c_frame.data[pixel_offset + 2] = (unsigned char)clamp(class_colors[(int)mask.at<float>(i, j) % num_colors][2] * (1 - alpha) + c_frame.data[pixel_offset + 2] * alpha, 0, 255);  // b
            }
        }
        for (auto box : res.first)
            rectangle(c_frame, Point(box[0], box[1]), Point(box[2], box[3]), Scalar(0, 0, 255), 4);

        writer.write(c_frame);

        cout << count++ << endl;
    }
    detector.destroy();
    writer.release();
}