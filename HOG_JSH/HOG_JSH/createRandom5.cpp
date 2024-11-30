#include "hog.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>   // rand(), srand() 함수를 위해 추가
#include <ctime>     // time() 함수를 위해 추가
#include <algorithm> // std::shuffle을 위해 추가
#include <random>    // std::default_random_engine을 위해 추가

using namespace cv;
using namespace cv::ml;
using namespace std;

int main() {
    // 랜덤 시드 설정
    srand(static_cast<unsigned int>(time(0)));
    default_random_engine rng(static_cast<unsigned int>(time(0))); // 랜덤 엔진 생성

    // 클래스별 이미지 경로 설정
    vector<String> class0Paths;
    vector<String> class1Paths;
    glob("traindata/true/*", class0Paths, false);
    glob("traindata/false/*", class1Paths, false);

    // HOG 인스턴스 생성 및 초기화
    Hog hog;
    hog.setCellSize(8);  // 셀 크기 설정
    hog.setBlockSize(2); // 블록 크기 설정

    // 10개의 SVM 모델을 생성 및 학습
    for (int modelIndex = 0; modelIndex < 5; ++modelIndex) {
        vector<pair<String, int>> imagePathsAndLabels;

        // 클래스 0 이미지 경로와 라벨(1) 추가 (true: 1)
        for (const auto& path : class0Paths) {
            imagePathsAndLabels.push_back(make_pair(path, 1)); // true를 1로 라벨링
        }

        // 클래스 1 이미지 경로와 라벨(-1) 추가 (false: -1)
        for (const auto& path : class1Paths) {
            imagePathsAndLabels.push_back(make_pair(path, -1)); // false를 -1로 라벨링
        }

        // 이미지 경로와 라벨을 std::shuffle로 무작위로 섞음
        shuffle(imagePathsAndLabels.begin(), imagePathsAndLabels.end(), rng);

        Mat trainingData;
        vector<int> labels;

        // 무작위로 섞인 이미지 경로와 라벨을 사용하여 특징 추출 및 학습 데이터 구성
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
        for (size_t i = 0; i < labels.size(); i++) {
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
            string modelPath = "svm_model_false" + to_string(modelIndex) + ".xml";
            svm->save(modelPath);
            cout << "SVM training for model " << modelIndex << " completed and model saved as " << modelPath << endl;
        }
        else {
            cerr << "Error: SVM model " << modelIndex << " was not trained." << endl;
        }
    }

    return 0;
}
