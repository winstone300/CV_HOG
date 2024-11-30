#include "hog.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>   // rand(), srand() 함수를 위해 추가
#include <ctime>     // time() 함수를 위해 추가

using namespace cv;
using namespace cv::ml;
using namespace std;

int main() {
    // 클래스별 이미지 경로 설정
    vector<String> class0Paths;
    vector<String> class1Paths;
    glob("traindata/true/*", class0Paths, false);
    glob("traindata/false/*", class1Paths, false);

    // 이미지 경로와 라벨을 저장할 벡터 생성
    vector<pair<String, int>> imagePathsAndLabels;

    // 두 클래스의 이미지 개수 중 작은 값을 찾아 반복 수를 결정
    size_t minSize = min(class0Paths.size(), class1Paths.size());
    size_t maxSize = max(class0Paths.size(), class1Paths.size());

    // 번갈아가며 true(1)와 false(-1) 데이터를 추가
    for (size_t i = 0; i < minSize; ++i) {
        imagePathsAndLabels.push_back(make_pair(class0Paths[i], 1));  // true를 1로 라벨링
        imagePathsAndLabels.push_back(make_pair(class1Paths[i], -1)); // false를 -1로 라벨링
    }

    // 만약 class0Paths나 class1Paths에 더 많은 데이터가 있다면 남은 데이터를 추가
    if (class0Paths.size() > class1Paths.size()) {
        for (size_t i = minSize; i < class0Paths.size(); ++i) {
            imagePathsAndLabels.push_back(make_pair(class0Paths[i], 1));
        }
    }
    else if (class1Paths.size() > class0Paths.size()) {
        for (size_t i = minSize; i < class1Paths.size(); ++i) {
            imagePathsAndLabels.push_back(make_pair(class1Paths[i], -1));
        }
    }

    // HOG 인스턴스 생성 및 초기화
    Hog hog;
    hog.setCellSize(8);  // 셀 크기 설정
    hog.setBlockSize(2); // 블록 크기 설정

    Mat trainingData;
    vector<int> labels;

    // 이미지 경로와 라벨을 사용하여 특징 추출 및 학습 데이터 구성
    for (const auto& item : imagePathsAndLabels) {
        String imagePath = item.first;
        int label = item.second;

        Mat img = imread(imagePath, IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "Error: Could not load image: " << imagePath << endl;
            continue;
        }

        resize(img, img, Size(64, 128));

        Mat angle(img.rows, img.cols, CV_64F);
        Mat scalar(img.rows, img.cols, CV_64F);
        unsigned char* imgData = img.data;
        hog.RunSobelFilter(imgData, angle, scalar, img.cols, img.rows, 3, 1, 1);

        vector<vector<double>> allHisto = hog.getAllHisto(imgData, angle, scalar, img.cols, img.rows);
        vector<double> featureVector = hog.normalize(allHisto, img.cols, img.rows);

        if (featureVector.empty()) {
            cerr << "Error: Feature vector is empty for image: " << imagePath << endl;
            continue;
        }

        Mat featureMat(1, featureVector.size(), CV_32F);
        for (size_t j = 0; j < featureVector.size(); ++j) {
            featureMat.at<float>(0, j) = static_cast<float>(featureVector[j]);
        }

        trainingData.push_back(featureMat);
        labels.push_back(label);
    }

    // 라벨을 Mat 형식으로 변환
    Mat labelsMat(labels.size(), 1, CV_32S);
    for (size_t i = 0; i < labels.size(); ++i) {
        labelsMat.at<int>(i, 0) = labels[i];
    }

    // SVM 설정 및 학습
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);        // 서포트 벡터 분류기
    svm->setKernel(SVM::LINEAR);     // 선형 커널 사용
    svm->setC(1.0);                  // 오류에 대한 패널티

    // SVM 학습
    Ptr<TrainData> trainData = TrainData::create(trainingData, ROW_SAMPLE, labelsMat);
    svm->train(trainData);

    // SVM 모델 저장
    if (svm->isTrained()) {
        svm->save("svm_model07.xml");
        cout << "SVM training completed and model saved!" << endl;
    }
    else {
        cerr << "Error: SVM model was not trained." << endl;
    }

    return 0;
}
