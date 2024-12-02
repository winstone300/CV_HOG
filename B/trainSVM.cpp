#include "hog.h"    //구현한 hog include
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>  
#include <ctime>    

using namespace cv;
using namespace cv::ml;
using namespace std;

int main() {
    srand(static_cast<unsigned int>(time(0)));

    vector<String> path1;
    vector<String> path2;
    glob("traindata/trainTrue/*", path1, false);
    glob("traindata/false/*", path2, false);

    vector<pair<String, int>> imgData;

    for (int i = 0; i < path1.size(); i++) {
        imgData.push_back(make_pair(path1[i], 1)); // true = 1    
    }
    for (int i = 0; i < path2.size(); i++) {
        imgData.push_back(make_pair(path2[i], -1)); // false = -1
    }

    //random하게 svm에 학습시키기 위해 random하게 swap하여 순서를 섞어줌
    for (size_t i = imgData.size() - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        swap(imgData[i], imgData[j]);
    }

    //hog객체 생성
    Hog hog;
    hog.setCellSize(8);
    hog.setBlockSize(2);

    Mat trainingData;
    vector<int> labels;

    for (int k = 0; k < imgData.size(); k++) {
        String imagePath = imgData[k].first;
        int label = imgData[k].second;

        Mat img = imread(imagePath, IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "Error: Could not load image: " << imagePath << endl;
            continue;
        }

        resize(img, img, Size(64, 128));

        // 특징벡터 추출
        vector<double> featureVector = hog.getFeature(img);

        if (featureVector.empty()) {
            cerr << "Error: Feature vector is empty for image: " << imagePath << endl;
            continue;
        }

        // double형태로 특징벡터를 구현하여 svm에 맞게 float타입으로 바꿔줌
        Mat featureMat(1, featureVector.size(), CV_32F);
        for (size_t j = 0; j < featureVector.size(); ++j) {
            featureMat.at<float>(0, j) = static_cast<float>(featureVector[j]);
        }

        trainingData.push_back(featureMat);
        labels.push_back(label);
    }

    // label Mat형식으로 넣어줌
    Mat labelMat(labels.size(), 1, CV_32S);
    for (size_t i = 0; i < labels.size(); ++i) {
        labelMat.at<int>(i, 0) = labels[i];
    }

    //svm학습
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setC(1.0);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));

    Ptr<TrainData> trainData = TrainData::create(trainingData, ROW_SAMPLE, labelMat);
    svm->train(trainData);

    if (svm->isTrained()) {
        svm->save("svm_model_newFoldT.xml");
        cout << "SVM training completed and model saved!" << endl;
    }
    else {
        cerr << "Error: SVM model was not trained." << endl;
    }

    return 0;
}