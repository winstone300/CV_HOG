#pragma once

#include <opencv2/opencv.hpp>

namespace noDup {
	void hogFeatVec(const cv::Mat& in, cv::Mat& out);
	void hog(const cv::Mat& in, cv::Mat& out);
	void padding(const cv::Mat& in, cv::Mat& out, int padSize);
	void gradient(const cv::Mat& in, cv::Mat& out, int padSize);
	void histogramGradient(const cv::Mat& in, cv::Mat& out, int cellSize);
	void histNormalize(const cv::Mat& in, cv::Mat& out);
}