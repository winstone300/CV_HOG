#include <opencv2/opencv.hpp>

#include <cmath>
#include <vector>

#define PI	(3.14159265358979323846)

//#define SHOW_GRADIENT

using namespace cv;

void padding(const Mat& in, Mat& out, int padSize);
void gradient(const Mat& in, Mat& out, int padSize);
void histogramGradient(const Mat& in, Mat& out, int cellSize);
void histNormalize(Mat& in, Mat& out);

int main() {
	Mat image = imread("Lena.png");
	Mat padImage, gradientImage, histImage, normalizeHistImage;

	padding(image, padImage, 1);
	gradient(padImage, gradientImage, 1);
	histogramGradient(gradientImage, histImage, 8);
	histNormalize(histImage, normalizeHistImage);

	imshow("img", image);

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
				degree = atan2(gY, gX) * 180 / PI;
				angle.data[channel * (y * angle.cols + x) + c] = (unsigned char)(degree < 0 ? degree + 180 : degree) % 180;
			}
		}
	}

	// 채널 개수 생각
	for (int y = 0; y < row; y++) {
		for (int x = 0; x < col; x++) {
			int c = -1;
			int b = magnitude.data[channel * (y * magnitude.cols + x) + 0];
			int g = magnitude.data[channel * (y * magnitude.cols + x) + 1];
			int r = magnitude.data[channel * (y * magnitude.cols + x) + 2];

			if (b > g) {
				if (b > r) { // b
					c = 0;
				} else { // r
					c = 2;
				}
			} else {
				if (g > r) { // g
					c = 1;
				} else { // r
					c = 2;
				}
			}

			if (c == -1) {
				std::cerr << "gradient - max magnitude value is not found\n";
				return;
			}

			out.data[2 * (y * out.cols + x) + 0] = magnitude.data[channel * (y * magnitude.cols + x) + c];
			out.data[2 * (y * out.cols + x) + 1] = angle.data[channel * (y * angle.cols + x) + c];
		}
	}

#ifdef SHOW_GRADIENT
	imshow("gx", gx);
	imshow("gy", gy);
	imshow("magnitude", magnitude);
	imshow("angle", angle);
	waitKey();
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

			float norm = sqrt(sum);

			float* outPtr1 = out.ptr<float>(row);
			float* outPtr2 = out.ptr<float>(row + 1);

			for (int i = 0; i < 9; i++) {
				outPtr1[col * 9 + i] += histogram[i] / norm;
				outPtr1[col * 9 + i + 9] += histogram[i + 9] / norm;
				outPtr2[col * 9 + i] += histogram[i + 18] / norm;
				outPtr2[col * 9 + i + 9] += histogram[i + 27] / norm;
			}
		}
	}

	for (int row = 0; row < out.rows; row++) {
		float* outPtr = out.ptr<float>(row);

		for (int col = 0; col < out.cols; col++) {
			if ((row == 0 || row == out.rows - 1) && (col == 0 || col == out.cols - 1)) {
				break;
			} else if (row == 0 || row == out.rows - 1 || col == 0 || col == out.cols - 1) {
				for (int i = 0; i < 9; i++) {
					outPtr[col * 9 + i] /= 2;
				}
			} else {
				for (int i = 0; i < 9; i++) {
					outPtr[col * 9 + i] /= 4;
				}
			}
		}
	}
}

// Mat 타입
// 정규화 할때 중복되는 부분