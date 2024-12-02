#include "hog.h"    //구현한 hog헤더파일 include
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <unordered_set>

using namespace cv;
using namespace cv::ml;
using namespace std;

int main() {
    //svm모델 가져오기
    Ptr<SVM> svm = SVM::load("svm_model_newFoldT.xml");
    if (svm.empty()) {
        cerr << "Error: Could not load SVM model." << endl;
        return -1;
    }

    vector<String> truePath;
    vector<String> falsePath;
    // test이미지 경로
    glob("traindata/test/trueTest/*", truePath, false);
    glob("traindata/test/falseTest/*", falsePath, false);

    cout << "Total number of trueTest images: " << truePath.size() << endl;
    cout << "Total number of falseTest images: " << falsePath.size() << endl;

    Hog hog;
    hog.setCellSize(8);
    hog.setBlockSize(2);

    unordered_set<String> truePositiveImages;
    unordered_set<String> falsePositiveImages;

    for (size_t i = 0; i < truePath.size(); i++) {
        const String& imagePath = truePath[i];
        Mat img = imread(imagePath, IMREAD_GRAYSCALE);       
        if (img.empty()) {
            cerr << "Error: Could not load image"<< endl;
            continue;
        }

        resize(img, img, Size(64, 128));

        vector<double> featureVector = hog.getFeature(img);

        Mat featureMat(1, static_cast<int>(featureVector.size()), CV_32F);
        for (size_t j = 0; j < featureVector.size(); j++) {
            featureMat.at<float>(0, static_cast<int>(j)) = static_cast<float>(featureVector[j]);
        }

        float response = svm->predict(featureMat);
        int predictedLabel = static_cast<int>(response);

        if (predictedLabel == 1) {
            truePositiveImages.insert(imagePath); // TP
        }
        // fn이미지 저장
        else {
            String outputPath = "fn/" + imagePath.substr(imagePath.find_last_of("/\\") + 1);
            imwrite(outputPath, img);
        }
    }

    for (size_t i = 0; i < falsePath.size(); i++) {
        const String& imagePath = falsePath[i];
        Mat img = imread(imagePath, IMREAD_GRAYSCALE);

        resize(img, img, Size(64, 128));

        vector<double> featureVector = hog.getFeature(img);


        Mat featureMat(1, static_cast<int>(featureVector.size()), CV_32F);
        for (size_t j = 0; j < featureVector.size(); j++) {
            featureMat.at<float>(0, static_cast<int>(j)) = static_cast<float>(featureVector[j]);
        }

        float response = svm->predict(featureMat);
        int predictedLabel = static_cast<int>(response);

        //FP이미지 저장
        if (predictedLabel == 1) {
            falsePositiveImages.insert(imagePath); // FP
            String outputPath = "fp/" + imagePath.substr(imagePath.find_last_of("/\\") + 1);
            imwrite(outputPath, img);
        }
    }

    cout << "\nTotal trueTest images processed: " << truePath.size() << endl;
    cout << "True Positives (TP): " << truePositiveImages.size() << endl;

    cout << "\nTotal falseTest images processed: " << falsePath.size() << endl;
    cout << "False Positives (FP): " << falsePositiveImages.size() << endl;

    return 0;
}
