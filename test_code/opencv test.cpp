#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

void opencv_objectDetection(); // opencv error rate
void detect(string); // opencv detection

int main() {
	opencv_objectDetection();

    string path = "input";

    for (const auto& entry : fs::directory_iterator(path)) {
        std::cout << entry.path().string() << std::endl;
        detect(entry.path().string());
    }

    return 0;
}

void opencv_objectDetection() {
	vector<String> testImagePaths;
	glob("input/*", testImagePaths, false);

	size_t totalImages = testImagePaths.size();
	std::cout << "Total number of images: " << totalImages << endl;

	Size windowSize(64, 128);

	// opencv HOG
	HOGDescriptor hog;
	hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

	// test
	int gWindow = 0;
	int gDetect = 0;

	for (size_t idx = 0; idx < testImagePaths.size(); ++idx) {
        std::cout << fs::path(testImagePaths[idx]).filename().string() << std::endl;
		Mat img = imread(testImagePaths[idx]);

		int stepSize = 16;

		//vector<double> scales = { 0.4, 0.8, 1.0, 1.2 };
		vector<double> scales = { 1.0 };

		// test
		int windowCount = 0;
		int detectCount = 0;

		for (double scale : scales) {
			Mat resizedImg;
			Size scaledSize(cvRound(img.cols * scale), cvRound(img.rows * scale));
			resize(img, resizedImg, scaledSize);

			if (resizedImg.cols < windowSize.width || resizedImg.rows < windowSize.height) {
				continue;
			}

			for (int y = 0; y <= resizedImg.rows - windowSize.height; y += stepSize) {
				for (int x = 0; x <= resizedImg.cols - windowSize.width; x += stepSize) {
					Rect windowRect(x, y, windowSize.width, windowSize.height);
					Mat window = resizedImg(windowRect);

					// save false data
					//string savePath = "temp/" + to_string(gWindow) + "_" + to_string(windowCount) + ".jpg";
					//imwrite(savePath, window);

					vector<Point> foundLocations;
					Mat descriptorMat;
					hog.detect(window, foundLocations);

					if (!foundLocations.empty()) {
						detectCount++; // test
					}
					windowCount++; // test
				}
			}
		}

		// test
		std::cout << "windowCount: " << windowCount << endl;
		std::cout << "detectCount: " << detectCount << endl;
		gWindow += windowCount;
		gDetect += detectCount;
	}

	// test
	std::cout << "gWindow: " << gWindow << endl;
	std::cout << "gDetect: " << gDetect << endl;
	std::cout << "error percent: " << 100.0 * gDetect / gWindow << endl;

	std::cout << "\nTotal images processed: " << totalImages << endl;
}

void detect(string filePath) {
    // 이미지 로드
    cv::Mat image = cv::imread(filePath);

    if (image.empty()) {
        std::cerr << "Image not loaded properly!" << std::endl;
        return;
    }

    // HOGDescriptor 객체 생성
    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    // 사람 검출
    std::vector<cv::Rect> detectedPersons;
    hog.detectMultiScale(image, detectedPersons, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);

    // 결과 출력
    for (size_t i = 0; i < detectedPersons.size(); ++i) {
        cv::rectangle(image, detectedPersons[i], cv::Scalar(0, 255, 0), 3);
    }

    string savePath = "output/" + fs::path(filePath).filename().string();
    imwrite(savePath, image);
}