#include <cmath>
#include <iostream>

#include "hog.hpp"

using namespace cv;

void noDup::hogFeatVec(const cv::Mat& in, cv::Mat& out) {
	Mat temp;

	hog(in, temp);

	out = temp.reshape(1, 1);
}

void noDup::hog(const Mat& in, Mat& out) {
	Mat padImage, gradientImage, histImage, normalizeHistImage;

	padding(in, padImage, 1);
	gradient(padImage, gradientImage, 1);
	histogramGradient(gradientImage, histImage, 8);
	histNormalize(histImage, normalizeHistImage);

	out = normalizeHistImage.clone();
}

void noDup::padding(const Mat& in, Mat& out, int padSize) {
	out = Mat::zeros(in.rows + 2 * padSize, in.cols + 2 * padSize, in.type());

	int channel = in.channels();

	for (int c = 0; c < in.channels(); c++) {

		for (int y = 0; y < in.rows; y++) {
			for (int x = 0; x < in.cols; x++) {
				out.data[channel * ((y + padSize) * out.cols + (x + padSize)) + c]
					= in.data[channel * (y * in.cols + x) + c];
			}
		}

		for (int y = 0; y < padSize; y++) {
			for (int x = 0; x < in.cols; x++) {
				out.data[channel * (y * out.cols + (x + padSize)) + c]
					= in.data[channel * x + c];
				out.data[channel * ((y + padSize + in.rows) * out.cols + (x + padSize)) + c]
					= in.data[channel * ((in.rows - 1) * in.cols + x) + c];
			}
		}

		for (int y = 0; y < out.rows; y++) {
			for (int x = 0; x < padSize; x++) {
				out.data[channel * (y * out.cols + x) + c]
					= out.data[channel * (y * out.cols + padSize) + c];
				out.data[channel * (y * out.cols + (x + padSize + in.cols)) + c]
					= out.data[channel * (y * out.cols + (padSize + in.cols - 1)) + c];
			}
		}
	}
}

void noDup::gradient(const Mat& in, Mat& out, int padSize) {
	int row = in.rows - 2 * padSize;
	int col = in.cols - 2 * padSize;
	int type = in.type();
	int channel = in.channels();

	out = Mat::zeros(row, col, CV_32FC2);
	//Mat gx = Mat::zeros(row, col, type);
	//Mat gy = Mat::zeros(row, col, type);
	Mat magnitude = Mat::zeros(row, col, CV_32FC(channel));
	Mat angle = Mat::zeros(row, col, CV_32FC(channel));

	float* outPtr = out.ptr<float>();
	float* magPtr = magnitude.ptr<float>();
	float* angPtr = angle.ptr<float>();

	for (int c = 0; c < channel; c++) {

		int gX, gY;
		float degree;

		for (int y = 0; y < row; y++) {
			for (int x = 0; x < col; x++) {
				gX = in.data[channel * ((y + padSize) * in.cols + (x + padSize + 1)) + c]
					- in.data[channel * ((y + padSize) * in.cols + (x + padSize - 1)) + c];
				gY = in.data[channel * ((y + padSize + 1) * in.cols + (x + padSize)) + c]
					- in.data[channel * ((y + padSize - 1) * in.cols + (x + padSize)) + c];

				//gx.data[channel * (y * gx.cols + x) + c] = gX;
				//gy.data[channel * (y * gy.cols + x) + c] = gY;

				//mag = sqrt(gX * gX + gY * gY);
				magPtr[channel * (y * magnitude.cols + x) + c] = (float)sqrt(gX * gX + gY * gY);
				degree = (float)(atan2(gY, gX) * 180 / CV_PI) + 180;
				angPtr[channel * (y * angle.cols + x) + c] = (float)fmod(degree, 180);
				//angle.data[channel * (y * angle.cols + x) + c] = (unsigned char)(degree < 0 ? degree + 180 : degree) % 180;
			}
		}
	}

	float* pixelValue = new float[channel];
	int cIndex;
	float maxValue;

	for (int y = 0; y < row; y++) {
		for (int x = 0; x < col; x++) {
			cIndex = -1;
			maxValue = -1;

			for (int c = 0; c < channel; c++) {
				pixelValue[c] = magPtr[channel * (y * magnitude.cols + x) + c];
			}

			for (int c = 0; c < channel; c++) {
				if (pixelValue[c] > maxValue) {
					maxValue = pixelValue[c];
					cIndex = c;
				}
			}

			if (cIndex == -1 || maxValue == -1) {
				std::cerr << "gradient - max magnitude value is not found\n";
				return;
			}

			outPtr[2 * (y * out.cols + x) + 0] = magPtr[channel * (y * magnitude.cols + x) + cIndex];
			outPtr[2 * (y * out.cols + x) + 1] = angPtr[channel * (y * angle.cols + x) + cIndex];
		}
	}

	delete[] pixelValue;

#pragma region deprecated_colorGradient
	// only color mat
	//for (int y = 0; y < row; y++) {
	//	for (int x = 0; x < col; x++) {
	//		int c = -1;
	//		int b = magnitude.data[channel * (y * magnitude.cols + x) + 0];
	//		int g = magnitude.data[channel * (y * magnitude.cols + x) + 1];
	//		int r = magnitude.data[channel * (y * magnitude.cols + x) + 2];

	//		if (b > g) {
	//			if (b > r) { // b
	//				c = 0;
	//			} else { // r
	//				c = 2;
	//			}
	//		} else {
	//			if (g > r) { // g
	//				c = 1;
	//			} else { // r
	//				c = 2;
	//			}
	//		}

	//		if (c == -1) {
	//			std::cerr << "gradient - max magnitude value is not found\n";
	//			return;
	//		}

	//		out.data[2 * (y * out.cols + x) + 0] = magnitude.data[channel * (y * magnitude.cols + x) + c];
	//		out.data[2 * (y * out.cols + x) + 1] = angle.data[channel * (y * angle.cols + x) + c];
	//	}
	//}
#pragma endregion
}

void noDup::histogramGradient(const Mat& in, Mat& out, int cellSize) {
	if (in.type() != CV_32FC2) {
		std::cerr << "histogramGradient - input Mat type error\n";
		return;
	} else if (in.rows % cellSize != 0 || in.cols % cellSize != 0) {
		std::cerr << "histogramGradient - input Mat size error\n";
		return;
	}

	int rowCell = in.rows / cellSize, colCell = in.cols / cellSize;
	const float* inPtr = in.ptr<float>();

	//out = Mat::zeros(rowCell, colCell, CV_16UC(9));
	out = Mat::zeros(rowCell, colCell, CV_32FC(9));

	for (int row = 0; row < rowCell; row++) {
		//ushort* rowPtr = out.ptr<ushort>(row);
		float* rowPtr = out.ptr<float>(row);

		for (int col = 0; col < colCell; col++) {
			Point start(col * cellSize, row * cellSize);

			for (int y = 0; y < cellSize; y++) {
				for (int x = 0; x < cellSize; x++) {
					float pixelMag = inPtr[2 * ((start.y + y) * in.cols + (start.x + x)) + 0];
					float pixelAngle = inPtr[2 * ((start.y + y) * in.cols + (start.x + x)) + 1];

					int histIndex = (int)pixelAngle / 20;
					float rate = (float)fmod(pixelAngle, 20) / 20;

					if (histIndex < 0 || histIndex > 8) {
						std::cerr << "histogramGradient - histogram index error\n";
						return;
					}

					rowPtr[col * 9 + histIndex] += pixelMag * (1 - rate);
					rowPtr[col * 9 + (histIndex + 1) % 9] += pixelMag * rate;
				}
			}
		}
	}
}

void noDup::histNormalize(const Mat& in, Mat& out) {
	if (in.type() != CV_32FC(9)) {
		std::cerr << "histNormalization - input Mat type error\n";
		return;
	}

	out = Mat::zeros(in.rows, in.cols, CV_32FC(9));

	float histogram[36] = { 0, };

	for (int row = 0; row < in.rows - 1; row++) {
		for (int col = 0; col < in.cols - 1; col++) {
			const float* inPtr1 = in.ptr<float>(row);
			const float* inPtr2 = in.ptr<float>(row + 1);

			for (int i = 0; i < 18; i++) {
				histogram[i] = inPtr1[col * 9 + i];
				histogram[i + 18] = inPtr2[col * 9 + i];
			}

			float sum = 0;
			for (int i = 0; i < 36; i++) {
				sum += histogram[i] * histogram[i];
			}
			sum = (sum == 0) ? 1 : sum;

			if (sum == 0) {
				std::cerr << "histNormalization - sum value error\n";
				return;
			}

			float norm = sqrt(sum);

			float* outPtr1 = out.ptr<float>(row);
			float* outPtr2 = out.ptr<float>(row + 1);

			for (int i = 0; i < 18; i++) {
				outPtr1[col * 9 + i] += histogram[i] / norm;
				outPtr2[col * 9 + i] += histogram[i + 18] / norm;
			}
		}
	}

	for (int row = 0; row < out.rows; row++) {
		float* outPtr = out.ptr<float>(row);

		for (int col = 0; col < out.cols; col++) {
			if ((row == 0 || row == out.rows - 1) && (col == 0 || col == out.cols - 1)) {
				continue; // vertex
			} else if (row == 0 || row == out.rows - 1 || col == 0 || col == out.cols - 1) {
				for (int i = 0; i < 9; i++) {
					outPtr[col * 9 + i] /= 2; // edge
				}
			} else {
				for (int i = 0; i < 9; i++) {
					outPtr[col * 9 + i] /= 4; // inside
				}
			}
		}
	}
}
