#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
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
    glob("traindata/true/*", path1, false);
    glob("traindata/false_old/*", path2, false);

    vector<pair<String, int>> imgData;

    for (int i = 0; i < path1.size(); i++) {
        imgData.push_back(make_pair(path1[i], 1)); // true = 1    
    }
    for (int i = 0; i < path2.size(); i++) {
        imgData.push_back(make_pair(path2[i], -1)); // false = -1
    }

    for (size_t i = imgData.size() - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        swap(imgData[i], imgData[j]);
    }

    // OpenCV HOGDescriptor ����
    HOGDescriptor hog(
        Size(64, 128),  // ������ ũ��
        Size(16, 16),   // ��� ũ��
        Size(8, 8),     // ��� ����
        Size(8, 8),     // �� ũ��
        9               // ���� ��
    );

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

        vector<float> featureVector;
        hog.compute(img, featureVector);

        if (featureVector.empty()) {
            cerr << "Error: Feature vector is empty for image: " << imagePath << endl;
            continue;
        }

        Mat featureMat(1, featureVector.size(), CV_32F);
        for (size_t j = 0; j < featureVector.size(); ++j) {
            featureMat.at<float>(0, j) = featureVector[j];
        }

        trainingData.push_back(featureMat);
        labels.push_back(label);
    }

    Mat labelMat(labels.size(), 1, CV_32S);
    for (size_t i = 0; i < labels.size(); ++i) {
        labelMat.at<int>(i, 0) = labels[i];
    }

    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setC(1.0);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));

    Ptr<TrainData> trainData = TrainData::create(trainingData, ROW_SAMPLE, labelMat);
    svm->train(trainData);

    if (svm->isTrained()) {
        svm->save("svm_OpenCV_HOG.xml");
        cout << "SVM training completed and model saved!" << endl;
    }
    else {
        cerr << "Error: SVM model was not trained." << endl;
    }

    return 0;
}
