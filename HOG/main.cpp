#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <random>

#include "hog.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;
namespace fs = std::filesystem;

void objectDetect1(const Mat& img);
void objectDetect2(const Mat& testImage);
void svmTrain();
void svmPred();

int countFilesInDirectory(const std::string& folderPath);
void addGaussianNoise(const cv::Mat& src, cv::Mat& dst, double mean, double stddev);

int main() {
	//svmTrain();
	//svmPred();
	//return 0;

	Mat temp;

	objectDetect2(temp);

	return 0;

	//VideoCapture video("vtest.avi");

	//int index = 0;

	//while (1) {
	//	cout << "frame: " << index++ << "\n";
	//	Mat frame;
	//	video >> frame;

	//	if (frame.empty()) {
	//		break;
	//	}

	//	objectDetect2(frame);
	//}
}

// 내가 만들었던거
void objectDetect1(const Mat& img) {
	if (img.empty()) {
		std::cerr << "Error loading image" << std::endl;
		return;
	}

	// SVM 모델 불러오기
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("trained_svm.xml");
	if (svm->empty()) {
		std::cerr << "Error loading SVM model" << std::endl;
		return;
	}

	std::vector<cv::Rect> detections;

	double scale = 1.0;
	Mat resizedFrame = img.clone();
	while (resizedFrame.cols >= 64 && resizedFrame.rows >= 128) {
		std::cout << resizedFrame.size() << ' ' << (resizedFrame.rows - 128) / 5 * (resizedFrame.cols - 64) / 5 << '\n';
		// 슬라이딩 윈도우를 적용하여 HOG 특징 추출
		for (int y = 0; y < resizedFrame.rows - 128; y += 5) {
			for (int x = 0; x < resizedFrame.cols - 64; x += 5) {
				cv::Rect roi(x, y, 64, 128);
				cv::Mat patch = resizedFrame(roi);

				// HOG 특징 벡터 계산
				Mat descriptorMat;
				noDup::hogFeatVec(patch, descriptorMat);

				// SVM 예측
				float response = svm->predict(descriptorMat);
				if (response == 1) {
					// 원본 이미지의 크기에 맞게 박스 크기 조정
					cv::Rect originalSizeBox(cvRound(x * scale), cvRound(y * scale), cvRound(64 * scale), cvRound(128 * scale));
					detections.push_back(originalSizeBox);
				}
			}
		}

		// 이미지 크기 줄이기
		scale *= 1.5;
		cv::resize(resizedFrame, resizedFrame, cv::Size(), 1.0 / 1.2, 1.0 / 1.2);
	}

	for (const auto& rect : detections) {
		cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);
	}

	// 결과 화면에 표시
	cv::imshow("Person Detection", img);

	cv::waitKey();
}

void objectDetect2(const Mat& testImage2) {
	Ptr<SVM> svm = SVM::load("trained_svm.xml");

	string path = "dataset/detect";
	for (const auto& entry : fs::directory_iterator(path)) {
		cout << '\n' << entry.path().string() << '\n';

		double scaleFactor = 1.2;
		double minScale = 0.7;  // 최소 스케일 (이미지 축소)
		double maxScale = 1.3;  // 최대 스케일 (이미지 확대)

		Mat testImage = imread(entry.path().string());

		if (testImage.empty()) {
			cout << "Can't Open file" << endl;
			//continue;
		}

		vector<double> scales;
		// 스케일 다운 (이미지 축소)
		for (double scale = 1.0; scale >= minScale; scale /= scaleFactor) {
			scales.push_back(scale);
		}
		// 스케일 업 (이미지 확대)
		for (double scale = scaleFactor; scale <= maxScale; scale *= scaleFactor) {
			scales.push_back(scale);
		}

		vector<Rect> detectedRects;

		for (double currentScale : scales) {
			Mat resizeImage;
			resize(testImage, resizeImage, Size(), currentScale, currentScale);

			cout << resizeImage.size() << '\n';

			int height = resizeImage.rows;
			int width = resizeImage.cols;

			for (int h = 0; h < height - 128; h += 5) {
				for (int w = 0; w < width - 64; w += 5) {
					Rect range(w, h, 64, 128);
					Mat scr = resizeImage(range);

					Mat descriptorMat;
					noDup::hogFeatVec(scr, descriptorMat);

					float response = svm->predict(descriptorMat);
					if (response == 1) {
						int originalX = static_cast<int>(w / currentScale);
						int originalY = static_cast<int>(h / currentScale);
						int originalW = static_cast<int>(64 / currentScale);
						int originalH = static_cast<int>(128 / currentScale);

						detectedRects.push_back(Rect(originalX, originalY, originalW, originalH));
					}
				}
			}
		}

		vector<int> weights;
		groupRectangles(detectedRects, weights, 2, 0.3);

		for (const auto& rect : detectedRects) {
			rectangle(testImage, rect, Scalar(0, 255, 0), 2);
		}

		//cv::imshow("Person Detection", testImage);

		//cv::waitKey();

		string outputDir = "output";
		fs::create_directories(outputDir);
		string imageName = fs::path(entry.path()).filename().string();
		string savePath = outputDir + "/" + imageName;
		imwrite(savePath, testImage);
	}
}

void svmTrain() {
	std::string trainTrueFolder = "dataset/trainTrue";
	std::string trainFalseFolder = "dataset/trainFalse";

	Mat trainImg;
	std::vector<int> label;

	int fileCount = countFilesInDirectory(trainTrueFolder) + countFilesInDirectory(trainFalseFolder);
	int index = 0, percentage = 0;

	auto startHOG = std::chrono::high_resolution_clock::now();

	for (const auto& entry : fs::directory_iterator(trainTrueFolder)) {
		std::string filePath = entry.path().string();

		Mat img = imread(filePath, IMREAD_COLOR);

		if (img.rows != 128 || img.cols != 64) {
			resize(img, img, Size(64, 128));
		}

		if (!img.empty()) {
			Mat hogFeature;
			noDup::hogFeatVec(img, hogFeature);

			trainImg.push_back(hogFeature);
			label.push_back(1);

			//Mat imgF, hogFeatureF;
			//flip(img, imgF, 1);

			//dup0::HOG(imgF, hogFeatureF);

			//hogFeatureF = hogFeatureF.reshape(1, 1);

			//trainImg.push_back(hogFeatureF);
			//label.push_back(1);
		} else {
			std::cerr << filePath << " is not a valid image.\n";
		}

		if (++index >= (float)fileCount / 10 * percentage) {
			std::cout << percentage++ * 10 << "% 완료\n";
		} // 진행도 확인
	}

	for (const auto& entry : fs::directory_iterator(trainFalseFolder)) {
		std::string filePath = entry.path().string();

		Mat img = imread(filePath, IMREAD_COLOR);

		if (img.rows != 128 || img.cols != 64) {
			resize(img, img, Size(64, 128));
		}

		if (!img.empty()) {
			Mat hogFeature;

			noDup::hog(img, hogFeature);

			hogFeature = hogFeature.reshape(1, 1);

			trainImg.push_back(hogFeature);
			label.push_back(-1);
		} else {
			std::cerr << filePath << " is not a valid image.\n";
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
	std::vector<int> indices(trainImg.rows);
	for (int i = 0; i < trainImg.rows; ++i) {
		indices[i] = i;
	}
	std::cout << "shuffling...\n";
	// 인덱스를 무작위로 셔플링
	std::shuffle(indices.begin(), indices.end(), rng);

	// 셔플링된 인덱스를 사용하여 데이터와 라벨을 재배열
	cv::Mat shuffledData(trainImg.size(), trainImg.type());
	cv::Mat shuffledLabels(trainLabel.size(), trainLabel.type());

	for (int i = 0; i < indices.size(); ++i) {
		trainImg.row(indices[i]).copyTo(shuffledData.row(i));
		trainLabel.row(indices[i]).copyTo(shuffledLabels.row(i));
	}

	// create SVM
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-6));

	auto startSVM = std::chrono::high_resolution_clock::now();

	if (svm->train(shuffledData, cv::ml::ROW_SAMPLE, shuffledLabels)) {
		svm->save("trained_svm.xml");
		std::cout << "SVM 모델 학습 완료\n";
	} else {
		std::cerr << "SVM 학습 실패\n";
	}

	auto endSVM = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> durationSVM = endSVM - startSVM;

	std::cout << std::endl;
	std::cout << "학습 데이터 개수: " << fileCount << "\n";
	std::cout << "HOG 실행 시간: " << durationHOG.count() << "초\n";
	std::cout << "SVM 실행 시간: " << durationSVM.count() << "초\n\n";
}

void svmPred() {
	std::string testTrueFolder = "dataset/testTrue";
	std::string testFalseFolder = "dataset/testFalse";

	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("trained_svm.xml");
	if (svm.empty()) {
		std::cerr << "Error: Could not load the SVM model.\n";
		return;
	}

	int TP = 0, FN = 0, TN = 0, FP = 0;

	auto startPred = std::chrono::high_resolution_clock::now();

	// true test data
	for (const auto& entry : fs::directory_iterator(testTrueFolder)) {
		std::string filePath = entry.path().string();

		cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);

		if (img.rows != 128 || img.cols != 64) {
			resize(img, img, Size(64, 128));
		}

		if (!img.empty()) {
			cv::Mat hogFeature;

			noDup::hog(img, hogFeature);

			hogFeature = hogFeature.reshape(1, 1); // 행 벡터로 변환

			// 예측 수행
			float response = svm->predict(hogFeature);

			// 예측 결과 출력
			//std::cout << "File: " << filePath << " -> Prediction result: " << response << std::endl;

			response == 1 ? TP++ : FN++;
		} else {
			std::cerr << "Error: Could not load or process image: " << filePath << std::endl;
		}
	}

	// false test data
	for (const auto& entry : fs::directory_iterator(testFalseFolder)) {
		std::string filePath = entry.path().string();

		cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);

		if (img.rows != 128 || img.cols != 64) {
			resize(img, img, Size(64, 128));
		}

		if (!img.empty()) {
			cv::Mat hogFeature;

			noDup::hog(img, hogFeature);

			hogFeature = hogFeature.reshape(1, 1); // 행 벡터로 변환

			// 예측 수행
			float response = svm->predict(hogFeature);

			// 예측 결과 출력
			//std::cout << "File: " << filePath << " -> Prediction result: " << response << std::endl;

			//response == 1 ? TN++ : FP++;
			if (response == 1) {
				//std::cout << filePath << "\n";
				FP++;
			} else {
				TN++;
			}
		} else {
			std::cerr << "Error: Could not load or process image: " << filePath << std::endl;
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

void addGaussianNoise(const cv::Mat& src, cv::Mat& dst, double mean, double stddev) {
	cv::Mat noise = cv::Mat(src.size(), src.type());

	// 가우시안 노이즈 생성
	cv::randn(noise, mean, stddev);

	// 원본 영상에 노이즈 추가
	cv::add(src, noise, dst);

	// 결과 영상이 8비트일 경우 값 클리핑 (0 ~ 255)
	dst.convertTo(dst, CV_8U);
}