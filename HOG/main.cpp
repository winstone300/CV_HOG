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

void objectDetect(string filePath);

void svmTrain(string svm_name);
void svmPred(string svm_name);
void objectDetect1(const Mat& img);
void objectDetect2(const Mat& testImage);

int countFilesInDirectory(const string& folderPath);
void addGaussianNoise(const Mat& src, Mat& dst, double mean, double stddev);

bool MY_HOG = true; // false = opencv HOGDescriptor
bool TRAIN_FLIP = true;
bool SAVE_FALSE_PRED = false;
HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);

int main() {
	//svmTrain("new_trained_svm.xml");
	svmPred("trained_svm_my.xml");
	return 0;

	string path = "detection";

	for (const auto& entry : fs::directory_iterator(path)) {
		objectDetect(entry.path().string());
	}

	//objectDetect1(img);
	//objectDetect2(Mat());

	return 0;
}

void objectDetect(string filePath) {
	cout << filePath << endl;

	Mat image = imread(filePath, IMREAD_GRAYSCALE);
	Mat imageColor = imread(filePath);
	if (image.empty()) {
		cerr << "이미지를 불러올 수 없습니다!" << endl;
		return;
	}

	// 가우시안 피라미드 파라미터
	double scaleFactor = 1.25; // 피라미드 스케일
	int levels = 5; // 피라미드 레벨 수

	// SVM 및 HOG 설정
	Ptr<SVM> svm = SVM::load("trained_svm.xml"); // 학습된 SVM 모델 로드
	Size winSize(64, 128); // 윈도우 크기 (예: 사람 검출 기준)
	Size stepSize(8, 16); // 슬라이딩 윈도우 스텝 크기

	vector<Rect> allDetections;

	// 가우시안 피라미드를 사용하여 다중 스케일 검출 수행
	Mat currentImage = image.clone();
	for (int i = 0; i < levels; ++i) {
		cout << currentImage.size() << endl;
		// 객체 검출
		vector<Rect> detections;

		for (int y = 0; y <= image.rows - winSize.height; y += stepSize.height) {
			for (int x = 0; x <= image.cols - winSize.width; x += stepSize.width) {
				// 슬라이딩 윈도우 영역 추출
				Rect window(x, y, winSize.width, winSize.height);
				Mat roi = image(window);

				Mat descriptorMat;
				if (MY_HOG) {
					noDup::hogFeatVec(roi, descriptorMat);
				} else {
					vector<float> featureVec;
					hog.compute(roi, featureVec);
					Mat temp(featureVec);
					temp = temp.reshape(1, 1);
					descriptorMat = temp.clone();
				}

				float response = svm->predict(descriptorMat);

				if (response == 1) {
					detections.push_back(window);
				}
			}
		}

		// 원본 이미지 스케일로 변환 후 결과 저장
		for (auto& rect : detections) {
			Rect scaledRect(rect.x * pow(scaleFactor, i),
				rect.y * pow(scaleFactor, i),
				rect.width * pow(scaleFactor, i),
				rect.height * pow(scaleFactor, i));
			allDetections.push_back(scaledRect);
		}

		// 가우시안 피라미드 다운스케일
		resize(currentImage, currentImage, Size(), 1.0 / scaleFactor, 1.0 / scaleFactor);
	}

	vector<int> weights; // 각 사각형의 검출 빈도
	int groupThreshold = 5; // 최소 그룹화 임계값
	double eps = 0.2; // 크기 차이에 따른 그룹화 허용 오차
	groupRectangles(allDetections, weights, groupThreshold, eps);

	// 검출된 결과 출력
	for (auto& rect : allDetections) {
		rectangle(imageColor, rect, Scalar(0, 0, 255), 2);
	}

	string outputDir = "output";
	fs::create_directories(outputDir);
	string savePath = outputDir + "/" + fs::path(filePath).filename().string();
	imwrite(savePath, imageColor);

	// 결과 이미지 표시
	//imshow("Detections", imageColor);
	//waitKey(0);
}

void svmTrain(string svm_name) {
	string trainTrueFolder = "dataset/trainTrue";
	string trainFalseFolder = "dataset/trainFalse";

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
			cout << percentage++ * 10 << "% 완료\n";
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
			cout << percentage++ * 10 << "% 완료\n";
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
	cout << "shuffling...\n";
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
		cout << "SVM 모델 학습 완료\n";
	} else {
		cerr << "SVM 학습 실패\n";
	}

	auto endSVM = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> durationSVM = endSVM - startSVM;

	cout << endl;
	cout << "학습 데이터 개수: " << fileCount << "\n";
	cout << "양성 학습 데이터 개수: " << trueFileCount << "\n";
	cout << "좌우 반전 추가: " << (TRAIN_FLIP ? "true" : "false") << "\n";
	cout << "음성 학습 데이터 개수: " << falseFileCount << "\n";
	cout << "HOG 실행 시간: " << durationHOG.count() << "초\n";
	cout << "SVM 실행 시간: " << durationSVM.count() << "초\n\n";
}

void svmPred(string svm_name) {
	string testTrueFolder = "dataset/testTrue";
	string testFalseFolder = "dataset/testFalse";

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
			//cout << "File: " << filePath << " -> Prediction result: " << response << endl;

			//response == 1 ? TP++ : FN++;
			if (response == 1) {
				TP++;
			} else {
				if (SAVE_FALSE_PRED) {
					cout << filePath << "\n";
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
			//cout << "File: " << filePath << " -> Prediction result: " << response << endl;

			//response == 1 ? TN++ : FP++;
			if (response == 1) {
				if (SAVE_FALSE_PRED) {
					cout << filePath << "\n";
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

	cout << "TP: " << TP << "\n";
	cout << "FN: " << FN << "\n";
	cout << "TN: " << TN << "\n";
	cout << "FP: " << FP << "\n\n";
	cout << "SVM 예측 시간: " << durationPred.count() << "초\n\n";
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
		scale *= 1.2;
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

		double scaleFactor = 2;
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

		cv::imshow("Person Detection", testImage);

		cv::waitKey();

		//string outputDir = "output";
		//fs::create_directories(outputDir);
		//string imageName = fs::path(entry.path()).filename().string();
		//string savePath = outputDir + "/" + imageName;
		//imwrite(savePath, testImage);
	}
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