#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>
#include <filesystem>

using namespace cv;
using namespace std;

int main() {
    // OpenCV HOGDescriptor 생성 및 기본 탐지기 설정
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    vector<String> testImagePaths;
    glob("traindata/test/detect/*", testImagePaths, false);

    size_t totalImages = testImagePaths.size();
    cout << "Total number of images: " << totalImages << endl;

    string outputDir = "All_outputCVV";
    filesystem::create_directories(outputDir);

    for (size_t idx = 0; idx < testImagePaths.size(); ++idx) {
        Mat img = imread(testImagePaths[idx], IMREAD_GRAYSCALE);
        Mat imgColor = imread(testImagePaths[idx]);

        if (img.empty()) {
            cerr << "Error: Could not load image: " << testImagePaths[idx] << endl;
            continue;
        }

        vector<Rect> detections;
        vector<double> weights;

        // HOG 기반 다중 스케일 탐지
        hog.detectMultiScale(img, detections, weights, 0, Size(8, 8), Size(16, 16), 1.05, 2);

        if (!detections.empty()) {
            for (size_t i = 0; i < detections.size(); i++) {
                rectangle(imgColor, detections[i], Scalar(0, 0, 255), 2);
                cout << "Rectangle drawn at (" << detections[i].x << ", " << detections[i].y
                    << ") with size (" << detections[i].width << ", " << detections[i].height << ")" << endl;
            }

            string imageName = filesystem::path(testImagePaths[idx]).filename().string();
            string savePath = outputDir + "/" + imageName;
            imwrite(savePath, imgColor);

            cout << "Image saved with detections: " << savePath << endl;
        }
        else {
            cout << "No detections for image: " << testImagePaths[idx] << endl;
        }
    }

    cout << "\nTotal images processed: " << totalImages << endl;

    return 0;
}
