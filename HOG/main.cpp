#include <opencv2/opencv.hpp>

#include <cmath>

//#define SHOW_PADDING
//#define SHOW_GRADIENT
//#define COUT_NORMALIZE_VALUE_RANGE

using namespace cv;

void padding(const Mat& in, Mat& out, int padSize);
void gradient(const Mat& in, Mat& out, int padSize);
void histogramGradient(const Mat& in, Mat& out, int cellSize);
void histNormalize(Mat& in, Mat& out);

// 데카르트 좌표계 실험
void polarToCartesian(Mat& in, Mat& out);
Mat gammaCorrection(const cv::Mat& input, double gamma);

int main() {
	Mat image = imread("Lena.png");
	Mat resize_image = image.clone();
	//resize(image, resize_image, Size(64, 128));
	Mat padImage, gradientImage, histImage, normalizeHistImage;

	padding(resize_image, padImage, 1);
	gradient(padImage, gradientImage, 1);
	histogramGradient(gradientImage, histImage, 8);
	histNormalize(histImage, normalizeHistImage);

#ifdef COUT_NORMALIZE_VALUE_RANGE
	float minValue = 1, maxValue = 0;
	for (int row = 0; row < normalizeHistImage.rows; row++) {
		float* ptr = normalizeHistImage.ptr<float>(row);
		for (int col = 0; col < normalizeHistImage.cols; col++) {
			for (int i = 0; i < 9; i++) {
				if (ptr[col * 9 + i] <= 0) {
					std::cout << "row: " << row << "\tcol: " << col << "\tchannel: " << i << "\tvalue: " << ptr[col * 9 + i] << '\n';
				} else if (ptr[col * 9 + i] >= 1) {
					std::cout << "row: " << row << "\tcol: " << col << "\tchannel: " << i << "\tvalue: " << ptr[col * 9 + i] << '\n';
				}
				minValue = min(minValue, ptr[col * 9 + i]);
				maxValue = max(maxValue, ptr[col * 9 + i]);
			}
		}
	}
	std::cout << minValue << ' ' << maxValue << "\n\n";
#endif // COUT_NORMALIZE_VALUE_RANGE

	//Mat cartesianImage;

	//polarToCartesian(gradientImage, cartesianImage);

	//imshow("img", image);

	waitKey();

	return 0;
}

void padding(const Mat& in, Mat& out, int padSize) {
	out = Mat::zeros(in.rows + 2 * padSize, in.cols + 2 * padSize, in.type());

	int row = in.rows, col = in.cols;
	int channel = in.channels();

	for (int c = 0; c < in.channels(); c++) {

		for (int y = 0; y < in.rows; y++) {
			for (int x = 0; x < in.cols; x++) {
				out.data[channel * ((y + padSize) * (in.cols + 2 * padSize) + (x + padSize)) + c]
					= in.data[channel * (y * in.cols + x) + c];
			}
		}

		for (int y = 0; y < padSize; y++) {
			for (int x = 0; x < in.cols; x++) {
				out.data[channel * (y * (in.cols + 2 * padSize) + (x + padSize)) + c]
					= in.data[channel * x + c];
				out.data[channel * ((y + padSize + in.rows) * (in.cols + 2 * padSize) + (x + padSize)) + c]
					= in.data[channel * ((in.rows - 1) * in.cols + x) + c];
			}
		}

		for (int y = 0; y < in.rows + 2 * padSize; y++) {
			for (int x = 0; x < padSize; x++) {
				out.data[channel * (y * (in.cols + 2 * padSize) + x) + c]
					= out.data[channel * (y * (in.cols + 2 * padSize) + padSize) + c];
				out.data[channel * (y * (in.cols + 2 * padSize) + (x + padSize + in.cols)) + c]
					= out.data[channel * (y * (in.cols + 2 * padSize) + (padSize + in.cols - 1)) + c];
			}
		}
	}

#ifdef SHOW_PADDING
	imshow("padding", out);
	waitKey();
	destroyAllWindows();
#endif // SHOW_PADDING
}

void gradient(const Mat& in, Mat& out, int padSize) {
	int row = in.rows - 2 * padSize;
	int col = in.cols - 2 * padSize;
	int type = in.type();
	int channel = in.channels();

	out = Mat::zeros(row, col, CV_8UC2); // 타입 동적으로 조절?
	Mat gx = Mat::zeros(row, col, type);
	Mat gy = Mat::zeros(row, col, type);
	Mat magnitude = Mat::zeros(row, col, type);
	Mat angle = Mat::zeros(row, col, type);

	for (int c = 0; c < channel; c++) {

		int gX, gY;
		double degree;

		for (int y = 0; y < row; y++) {
			for (int x = 0; x < col; x++) {
				gX = in.data[channel * ((y + padSize) * in.cols + (x + padSize + 1)) + c]
					- in.data[channel * ((y + padSize) * in.cols + (x + padSize - 1)) + c];
				gY = in.data[channel * ((y + padSize + 1) * in.cols + (x + padSize)) + c]
					- in.data[channel * ((y + padSize - 1) * in.cols + (x + padSize)) + c];

				gx.data[channel * (y * gx.cols + x) + c] = gX;
				gy.data[channel * (y * gy.cols + x) + c] = gY;

				magnitude.data[channel * (y * magnitude.cols + x) + c] = (uchar)sqrt(gX * gX + gY * gY);
				degree = atan2(gY, gX) * 180 / CV_PI;
				angle.data[channel * (y * angle.cols + x) + c] = (unsigned char)(degree < 0 ? degree + 180 : degree) % 180;
			}
		}
	}

	int* pixelValue = new int[channel];
	int cIndex, maxValue;

	for (int y = 0; y < row; y++) {
		for (int x = 0; x < col; x++) {
			cIndex = -1;
			maxValue = -1;

			for (int c = 0; c < channel; c++) {
				pixelValue[c] = magnitude.data[channel * (y * magnitude.cols + x) + c];
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

			out.data[2 * (y * out.cols + x) + 0] = magnitude.data[channel * (y * magnitude.cols + x) + cIndex];
			out.data[2 * (y * out.cols + x) + 1] = angle.data[channel * (y * angle.cols + x) + cIndex];
		}
	}

	delete[] pixelValue;

#pragma region deprecated_colorGradient
	// 컬러 영상에서만 작동
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

#ifdef SHOW_GRADIENT
	imshow("gx", gx);
	imshow("gy", gy);
	imshow("magnitude", magnitude);
	imshow("angle", angle);
	waitKey();
	destroyAllWindows();
#endif // SHOW_GRADIENT
}

void histogramGradient(const Mat& in, Mat& out, int cellSize) {
	if (in.type() != CV_8UC2) {
		std::cerr << "histogramGradient - input Mat type error\n";
		return;
	} else if (in.rows % cellSize != 0 || in.cols % cellSize != 0) {
		std::cerr << "histogramGradient - input Mat size error\n";
		return;
	}

	int rowCell = in.rows / cellSize, colCell = in.cols / cellSize;

	out = Mat::zeros(rowCell, colCell, CV_16UC(9));
	//out = Mat::zeros(rowCell, colCell, CV_32FC(9));

	for (int row = 0; row < rowCell; row++) {
		ushort* rowPtr = out.ptr<ushort>(row);
		//double* rowPtr = out.ptr<double>(row);

		for (int col = 0; col < colCell; col++) {
			Point start(col * cellSize, row * cellSize);

			for (int y = 0; y < cellSize; y++) {
				for (int x = 0; x < cellSize; x++) {
					int pixelMag = in.data[2 * ((start.y + y) * in.cols + (start.x + x)) + 0];
					int pixelAngle = in.data[2 * ((start.y + y) * in.cols + (start.x + x)) + 1];

					int histIndex = pixelAngle / 20;
					double rate = pixelAngle % 20 / 20.0;

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

void histNormalize(Mat& in, Mat& out) {
	if (in.type() != CV_16UC(9)) {
		std::cerr << "histNormalization - input Mat type error\n";
		return;
	}

	out = Mat::zeros(in.rows, in.cols, CV_32FC(9));

	ushort histogram[36] = { 0, };

	for (int row = 0; row < in.rows - 1; row++) {
		for (int col = 0; col < in.cols - 1; col++) {
			ushort* inPtr1 = in.ptr<ushort>(row);
			ushort* inPtr2 = in.ptr<ushort>(row + 1);

			for (int i = 0; i < 18; i++) {
				histogram[i] = inPtr1[col * 9 + i];
				histogram[i + 18] = inPtr2[col * 9 + i];
			}

			float sum = 0;
			for (int i = 0; i < 36; i++) {
				sum += histogram[i] * histogram[i];
			}

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
				continue; // 꼭짓점
			} else if (row == 0 || row == out.rows - 1 || col == 0 || col == out.cols - 1) {
				for (int i = 0; i < 9; i++) {
					outPtr[col * 9 + i] /= 2; // 가장자리
				}
			} else {
				for (int i = 0; i < 9; i++) {
					outPtr[col * 9 + i] /= 4; // 내부
				}
			}
		}
	}
}

void polarToCartesian(Mat& in, Mat& out) {
	if (in.type() != CV_8UC2) {
		std::cerr << "polarToCartesian - input Mat type error\n";
		return;
	}

	int rows = in.rows, cols = in.cols, channel = in.channels();
	float xMax = 0, yMax = 0; // test

	out = Mat::zeros(in.rows, in.cols, CV_32FC2);

	for (int row = 0; row < rows; row++) {
		float* outPtr = out.ptr<float>(row);

		for (int col = 0; col < cols; col++) {
			int pixelMag = in.data[channel * (row * in.cols + col) + 0];
			int pixelAngle = in.data[channel * (row * in.cols + col) + 1];

			outPtr[col * channel + 0] = pixelMag * cos(pixelAngle); // x
			outPtr[col * channel + 1] = pixelMag * sin(pixelAngle); // y

			xMax = max(abs(outPtr[col * channel + 0]), xMax);
			yMax = max(abs(outPtr[col * channel + 1]), yMax);
		}
	}

	// test - 시각화

	xMax = floor(xMax + 0.5);
	yMax = floor(yMax + 0.5);

	Mat plane = Mat::zeros(2 * yMax + 2, 2 * xMax + 2, CV_8UC1);

	for (int row = 0; row < rows; row++) {
		float* outPtr = out.ptr<float>(row);

		for (int col = 0; col < cols; col++) {
			int xPixel = floor(outPtr[col * channel + 0] + 0.5); // x
			int yPixel = floor(outPtr[col * channel + 1] + 0.5); // y

			xPixel += plane.cols / 2;
			yPixel += plane.rows / 2;

			uchar* pixelPtr = plane.ptr<uchar>(plane.rows - yPixel);
			pixelPtr[xPixel]++;
		}
	}

	Mat planeGamma = gammaCorrection(plane, 0.1);

	imshow("plane", plane);
	imshow("gamma", planeGamma);
}

Mat gammaCorrection(const cv::Mat& input, double gamma) {
	// Look Up Table 생성
	cv::Mat lut(1, 256, CV_8UC1);
	for (int i = 0; i < 256; ++i) {
		lut.at<uchar>(i) = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
	}

	cv::Mat output;
	cv::LUT(input, lut, output); // LUT을 사용하여 변환
	return output;
}

// Mat 타입