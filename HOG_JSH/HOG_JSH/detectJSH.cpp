#include "hog.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/dnn.hpp> 
#include <iostream>
#include <vector>
#include <unordered_set>
#include <filesystem>

using namespace cv;
using namespace cv::ml;
using namespace std;

int main() {
    Ptr<SVM> svm = SVM::load("svm_model_aa.xml");
    if (svm.empty()) {
        cerr << "Error: Could not load SVM model." << endl;
        return -1;
    }
    if (!svm->isTrained()) {
        cerr << "Error: SVM model is not trained." << endl;
        return -1;
    }

    vector<String> testImagePaths;
    glob("traindata/test/detect/*", testImagePaths, false);

    size_t totalImages = testImagePaths.size();
    cout << "Total number of images: " << totalImages << endl;

    Hog hog;
    Size windowSize(64, 128);

    unordered_set<String> positiveImages;

    string outputDir = "outputJSH";
    filesystem::create_directories(outputDir);

    for (size_t idx = 0; idx < testImagePaths.size(); ++idx) {
        Mat img = imread(testImagePaths[idx], IMREAD_GRAYSCALE);
        Mat imgColor = imread(testImagePaths[idx]);

        vector<Rect> allPositiveWindows;
        vector<float> allScores; // SVM 결정 함수 값 저장

        int stepSize = 16;

        vector<double> scales = { 0.8, 1.0,1.2};

        for (double scale : scales) {
            Mat resizedImg, resizedImgColor;
            Size scaledSize(cvRound(img.cols * scale), cvRound(img.rows * scale));
            resize(img, resizedImg, scaledSize);
            resize(imgColor, resizedImgColor, scaledSize);

            if (resizedImg.cols < windowSize.width || resizedImg.rows < windowSize.height) {
                continue;
            }

            for (int y = 0; y <= resizedImg.rows - windowSize.height; y += stepSize) {
                for (int x = 0; x <= resizedImg.cols - windowSize.width; x += stepSize) {
                    Rect windowRect(x, y, windowSize.width, windowSize.height);
                    Mat window = resizedImg(windowRect);

                    vector<double> featureVector = hog.getFeature(window);

                    Mat featureMat(1, featureVector.size(), CV_32F);
                    for (size_t j = 0; j < featureVector.size(); j++) {
                        featureMat.at<float>(0, j) = static_cast<float>(featureVector[j]);
                    }

                    float response = svm->predict(featureMat, noArray(), StatModel::RAW_OUTPUT); 
                    if (response > 0) { // SVM 결정 함수 값이 0보다 작으면 긍정
                        Rect originalWindowRect(
                            cvRound(x / scale),
                            cvRound(y / scale),
                            cvRound(windowSize.width / scale),
                            cvRound(windowSize.height / scale)
                        );
                        allPositiveWindows.push_back(originalWindowRect);
                        allScores.push_back(response); // 결정 함수 값의 절댓값 저장 (값이 작을수록 신뢰도 높음)
                    }
                }
            }
        }

        vector<int> indices;
        if (!allPositiveWindows.empty()) {
            float scoreThreshold = 0.0f; // 모든 박스 고려
            float nmsThreshold = 0.1f;   // IOU 임계값 설정
            cv::dnn::NMSBoxes(allPositiveWindows, allScores, scoreThreshold, nmsThreshold, indices);

            vector<Rect> finalDetections;
            for (int idx : indices) {
                finalDetections.push_back(allPositiveWindows[idx]);
            }

            positiveImages.insert(testImagePaths[idx]);

            for (const auto& rect : finalDetections) {
                rectangle(imgColor, rect, Scalar(0, 0, 255), 2);
                cout << "Rectangle drawn at (" << rect.x << ", " << rect.y << ") with size (" << rect.width << ", " << rect.height << ")" << endl;
            }

            string imageName = filesystem::path(testImagePaths[idx]).filename().string();
            string savePath = outputDir + "/" + imageName;
            imwrite(savePath, imgColor);

            cout << "Image saved with detections: " << savePath << endl;
        }
    }

    cout << "\nTotal images processed: " << totalImages << endl;
    cout << "Total positive images detected: " << positiveImages.size() << endl;

    cout << "\nPositive Images Detected:" << endl;
    for (const auto& imagePath : positiveImages) {
        cout << "Image: " << imagePath << endl;
    }

    return 0;
}
