#include "hog.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>   // rand(), srand() �Լ��� ���� �߰�
#include <ctime>     // time() �Լ��� ���� �߰�
#include <algorithm> // std::shuffle�� ���� �߰�
#include <random>    // std::default_random_engine�� ���� �߰�

using namespace cv;
using namespace cv::ml;
using namespace std;

int main() {
    // ���� �õ� ����
    srand(static_cast<unsigned int>(time(0)));
    default_random_engine rng(static_cast<unsigned int>(time(0))); // ���� ���� ����

    // Ŭ������ �̹��� ��� ����
    vector<String> class0Paths;
    vector<String> class1Paths;
    glob("traindata/true/*", class0Paths, false);
    glob("traindata/false/*", class1Paths, false);

    // HOG �ν��Ͻ� ���� �� �ʱ�ȭ
    Hog hog;
    hog.setCellSize(8);  // �� ũ�� ����
    hog.setBlockSize(2); // ��� ũ�� ����

    // 10���� SVM ���� ���� �� �н�
    for (int modelIndex = 0; modelIndex < 5; ++modelIndex) {
        vector<pair<String, int>> imagePathsAndLabels;

        // Ŭ���� 0 �̹��� ��ο� ��(1) �߰� (true: 1)
        for (const auto& path : class0Paths) {
            imagePathsAndLabels.push_back(make_pair(path, 1)); // true�� 1�� �󺧸�
        }

        // Ŭ���� 1 �̹��� ��ο� ��(-1) �߰� (false: -1)
        for (const auto& path : class1Paths) {
            imagePathsAndLabels.push_back(make_pair(path, -1)); // false�� -1�� �󺧸�
        }

        // �̹��� ��ο� ���� std::shuffle�� �������� ����
        shuffle(imagePathsAndLabels.begin(), imagePathsAndLabels.end(), rng);

        Mat trainingData;
        vector<int> labels;

        // �������� ���� �̹��� ��ο� ���� ����Ͽ� Ư¡ ���� �� �н� ������ ����
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

        // ���� Mat �������� ��ȯ
        Mat labelsMat(labels.size(), 1, CV_32S);
        for (size_t i = 0; i < labels.size(); i++) {
            labelsMat.at<int>(i, 0) = labels[i];
        }

        // SVM ���� �� �н�
        Ptr<SVM> svm = SVM::create();
        svm->setType(SVM::C_SVC);        // ����Ʈ ���� �з���
        svm->setKernel(SVM::LINEAR);     // ���� Ŀ�� ���
        svm->setC(1.0);                  // ������ ���� �г�Ƽ

        // SVM �н�
        Ptr<TrainData> trainData = TrainData::create(trainingData, ROW_SAMPLE, labelsMat);
        svm->train(trainData);

        // SVM �� ����
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
