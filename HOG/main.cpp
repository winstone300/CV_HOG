#include <iostream>
#include <filesystem>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <random>

#include "hog.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;
namespace fs = std::filesystem;

void objectDetection();
void svmTrain(string svm_name);
void svmPred(string svm_name);

int countFilesInDirectory(const string& folderPath);

HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);

#pragma region Settings

bool MY_HOG = true; // false = opencv HOGDescriptor
bool TRAIN_FLIP = false;
bool SAVE_FALSE_PRED = false;
bool TEST_ERROR_DETECT = false;

#pragma endregion

string trainTrueFolder = "dataset/trainTrue";
string trainFalseFolder = "dataset/trainFalse";
string testTrueFolder = "dataset/testTrue";
string testFalseFolder = "dataset/testFalse";

int main() {
	svmTrain("trained_svm_my.xml");
	svmPred("trained_svm_my.xml");

	objectDetection();

	return 0;
}

void objectDetection() {
	Ptr<SVM> svm;

	if (MY_HOG) {
		std::cout << "MY HOG\n";
		svm = SVM::load("trained_svm_my.xml");
	} else {
		std::cout << "OPENCV HOG\n";
		svm = SVM::load("trained_svm_opencv.xml");
	}

	if (TEST_ERROR_DETECT) {
		std::cout << "test code" << std::endl;
	}

	if (svm.empty()) {
		cerr << "Error: Could not load SVM model." << endl;
		return;
	}
	if (!svm->isTrained()) {
		cerr << "Error: SVM model is not trained." << endl;
		return;
	}

	vector<String> testImagePaths;
	glob("detection/*", testImagePaths, false);

	size_t totalImages = testImagePaths.size();
	std::cout << "Total number of images: " << totalImages << endl;

	Size windowSize(64, 128);

	unordered_set<String> positiveImages;

	string outputDir = "output";
	filesystem::create_directories(outputDir);

	// opencv HOG
	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);

	// test
	int gWindow = 0;
	int gDetect = 0;

	for (size_t idx = 0; idx < testImagePaths.size(); ++idx) {
		Mat img = imread(testImagePaths[idx], IMREAD_GRAYSCALE);
		Mat imgColor = imread(testImagePaths[idx]);

		vector<Rect> allPositiveWindows;
		vector<float> allScores; // SVM 결정 함수 값 저장

		int stepSize = 16;

		vector<double> scales = { 0.4, 0.8, 1.0, 1.2 };

		// test
		int windowCount = 0;
		int detectCount = 0;

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

					Mat descriptorMat;
					if (MY_HOG) {
						noDup::hogFeatVec(window, descriptorMat);
					} else {
						vector<float> featureVec;
						hog.compute(window, featureVec);
						Mat temp(featureVec);
						temp = temp.reshape(1, 1);
						descriptorMat = temp.clone();
					}

					float response = svm->predict(descriptorMat, noArray(), StatModel::RAW_OUTPUT);
					if (response > 0) {
						Rect originalWindowRect(
							cvRound(x / scale),
							cvRound(y / scale),
							cvRound(windowSize.width / scale),
							cvRound(windowSize.height / scale)
						);
						allPositiveWindows.push_back(originalWindowRect);
						allScores.push_back(response);
						detectCount++; // test
					}
					windowCount++; // test
				}
			}
		}

		// test
		if (TEST_ERROR_DETECT) {
			std::cout << "windowCount: " << windowCount << endl;
			std::cout << "detectCount: " << detectCount << endl;
			gWindow += windowCount;
			gDetect += detectCount;
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
				//std::cout << "Rectangle drawn at (" << rect.x << ", " << rect.y << ") with size (" << rect.width << ", " << rect.height << ")" << endl;
			}

			string imageName = filesystem::path(testImagePaths[idx]).filename().string();
			string savePath = outputDir + "/" + imageName;
			imwrite(savePath, imgColor);
			std::cout << "Image saved with detections: " << savePath << endl;
		} else {
			string imageName = filesystem::path(testImagePaths[idx]).filename().string();
			string savePath = outputDir + "/" + imageName;
			imwrite(savePath, imgColor);
			std::cout << "Image saved with no detection: " << savePath << endl;
		}
	}

	// test
	if (TEST_ERROR_DETECT) {
		std::cout << "Window: " << gWindow << endl;
		std::cout << "Detect(Error): " << gDetect << endl;
		std::cout << "error percent: " << 100.0 * gDetect / gWindow << endl;
	}

	std::cout << "\nTotal images processed: " << totalImages << endl;
	std::cout << "Total positive images detected: " << positiveImages.size() << endl;

	std::cout << "\nPositive Images Detected:" << endl;
	for (const auto& imagePath : positiveImages) {
		std::cout << "Image: " << imagePath << endl;
	}
}

void svmTrain(string svm_name) {

	Mat trainImg;
	vector<int> label;

	int trueFileCount = countFilesInDirectory(trainTrueFolder);
	int falseFileCount = countFilesInDirectory(trainFalseFolder);
	int fileCount = trueFileCount + falseFileCount;
	int index = 0, percentage = 0;

	auto startHOG = std::chrono::high_resolution_clock::now();

	for (const auto& entry : fs::directory_iterator(trainTrueFolder)) {
		string filePath = entry.path().string();

		//Mat img = imread(filePath, IMREAD_COLOR);
		Mat img = imread(filePath, IMREAD_GRAYSCALE);

		if (img.rows != 128 || img.cols != 64) {
			resize(img, img, Size(64, 128));
		}

		if (!img.empty()) {
			if (MY_HOG) {
				Mat hogFeature;
				noDup::hogFeatVec(img, hogFeature);

				trainImg.push_back(hogFeature);
				label.push_back(1);
			} else {
				vector<float> featureVec;
				hog.compute(img, featureVec);
				Mat temp(featureVec);
				temp = temp.reshape(1, 1);
				trainImg.push_back(temp);
				label.push_back(1);
			}

			if (TRAIN_FLIP) { // train flip
				Mat imgF, hogFeatureF;
				flip(img, imgF, 1);
				noDup::hogFeatVec(imgF, hogFeatureF);

				trainImg.push_back(hogFeatureF);
				label.push_back(1);
			}
		} else {
			cerr << filePath << " is not a valid image.\n";
		}

		if (++index >= (float)fileCount / 10 * percentage) {
			std::cout << percentage++ * 10 << "% 완료\n";
		} // 진행도 확인
	}

	for (const auto& entry : fs::directory_iterator(trainFalseFolder)) {
		string filePath = entry.path().string();

		Mat img = imread(filePath, IMREAD_COLOR);

		if (img.rows != 128 || img.cols != 64) {
			resize(img, img, Size(64, 128));
		}

		if (!img.empty()) {
			if (MY_HOG) {
				Mat hogFeature;
				noDup::hogFeatVec(img, hogFeature);

				trainImg.push_back(hogFeature);
				label.push_back(-1);
			} else {
				vector<float> featureVec;
				hog.compute(img, featureVec);
				Mat temp(featureVec);
				temp = temp.reshape(1, 1);
				trainImg.push_back(temp);
				label.push_back(-1);
			}
		} else {
			cerr << filePath << " is not a valid image.\n";
		}

		if (++index >= (float)fileCount / 10 * percentage) {
			std::cout << percentage++ * 10 << "% 완료\n";
		} // 진행도 확인
	}

	auto endHOG = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> durationHOG = endHOG - startHOG;

	Mat trainLabel(label, true);
	trainLabel = trainLabel.reshape(1, label.size());
	trainLabel.convertTo(trainLabel, CV_32S);

	// random shuffling
	int seed = 42;  // 원하는 시드값
	std::mt19937 rng(seed);  // Mersenne Twister 엔진 사용

	// 인덱스 배열 생성
	vector<int> indices(trainImg.rows);
	for (int i = 0; i < trainImg.rows; ++i) {
		indices[i] = i;
	}
	std::cout << "shuffling...\n";
	// 인덱스를 무작위로 셔플링
	shuffle(indices.begin(), indices.end(), rng);

	// 셔플링된 인덱스를 사용하여 데이터와 라벨을 재배열
	Mat shuffledData(trainImg.size(), trainImg.type());
	Mat shuffledLabels(trainLabel.size(), trainLabel.type());

	for (int i = 0; i < indices.size(); ++i) {
		trainImg.row(indices[i]).copyTo(shuffledData.row(i));
		trainLabel.row(indices[i]).copyTo(shuffledLabels.row(i));
	}

	// create SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setC(1.0);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));

	auto startSVM = std::chrono::high_resolution_clock::now();

	if (svm->train(shuffledData, ROW_SAMPLE, shuffledLabels)) {
		svm->save(svm_name);
		std::cout << "SVM 모델 학습 완료\n";
	} else {
		cerr << "SVM 학습 실패\n";
	}

	auto endSVM = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> durationSVM = endSVM - startSVM;

	std::cout << endl;
	std::cout << "학습 데이터 개수: " << fileCount << "\n";
	std::cout << "양성 학습 데이터 개수: " << trueFileCount << "\t";
	std::cout << "좌우 반전 추가: " << (TRAIN_FLIP ? "true" : "false") << "\n";
	std::cout << "음성 학습 데이터 개수: " << falseFileCount << "\n";
	std::cout << "HOG 실행 시간: " << durationHOG.count() << "초\n";
	std::cout << "SVM 실행 시간: " << durationSVM.count() << "초\n\n";
}

void svmPred(string svm_name) {

	Ptr<SVM> svm = SVM::load(svm_name);
	if (svm.empty()) {
		cerr << "Error: Could not load the SVM model.\n";
		return;
	}

	int TP = 0, FN = 0, TN = 0, FP = 0;

	auto startPred = std::chrono::high_resolution_clock::now();

	// true test data
	for (const auto& entry : fs::directory_iterator(testTrueFolder)) {
		string filePath = entry.path().string();

		//Mat img = imread(filePath, IMREAD_COLOR);
		Mat img = imread(filePath, IMREAD_GRAYSCALE);

		if (img.rows != 128 || img.cols != 64) {
			resize(img, img, Size(64, 128));
		}

		if (!img.empty()) {
			Mat hogFeature;
			if (MY_HOG) {
				noDup::hogFeatVec(img, hogFeature);
			} else {
				vector<float> featureVec;
				hog.compute(img, featureVec);
				Mat temp(featureVec);
				temp = temp.reshape(1, 1);
				hogFeature = temp.clone();
			}

			// 예측 수행
			float response = svm->predict(hogFeature);

			// 예측 결과 출력
			//std::cout << "File: " << filePath << " -> Prediction result: " << response << endl;

			//response == 1 ? TP++ : FN++;
			if (response == 1) {
				TP++;
			} else {
				if (SAVE_FALSE_PRED) {
					std::cout << filePath << "\n";
					string savePath = "FN/" + fs::path(filePath).filename().string();
					imwrite(savePath, img);
				}
				FN++;
			}
		} else {
			cerr << "Error: Could not load or process image: " << filePath << endl;
		}
	}

	// false test data
	for (const auto& entry : fs::directory_iterator(testFalseFolder)) {
		string filePath = entry.path().string();

		Mat img = imread(filePath, IMREAD_COLOR);

		if (img.rows != 128 || img.cols != 64) {
			resize(img, img, Size(64, 128));
		}

		if (!img.empty()) {
			Mat hogFeature;
			if (MY_HOG) {
				noDup::hogFeatVec(img, hogFeature);
			} else {
				vector<float> featureVec;
				hog.compute(img, featureVec);
				Mat temp(featureVec);
				temp = temp.reshape(1, 1);
				hogFeature = temp.clone();
			}

			// 예측 수행
			float response = svm->predict(hogFeature);

			// 예측 결과 출력
			//std::cout << "File: " << filePath << " -> Prediction result: " << response << endl;

			//response == 1 ? TN++ : FP++;
			if (response == 1) {
				if (SAVE_FALSE_PRED) {
					std::cout << filePath << "\n";
					string savePath = "FP/" + fs::path(filePath).filename().string();
					imwrite(savePath, img);
				}
				FP++;
			} else {
				TN++;
			}
		} else {
			cerr << "Error: Could not load or process image: " << filePath << endl;
		}
	}

	auto endPred = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> durationPred = endPred - startPred;

	std::cout << "TP: " << TP << "\n";
	std::cout << "FN: " << FN << "\n";
	std::cout << "TN: " << TN << "\n";
	std::cout << "FP: " << FP << "\n\n";
	std::cout << "SVM 예측 시간: " << durationPred.count() << "초\n\n";
}

int countFilesInDirectory(const std::string& folderPath) {
	int fileCount = 0;

	// 폴더 내의 모든 항목을 순회
	for (const auto& entry : fs::directory_iterator(folderPath)) {
		// 디렉토리가 아닌 파일만 카운트
		if (fs::is_regular_file(entry.status())) {
			++fileCount;
		}
	}

	return fileCount;
}