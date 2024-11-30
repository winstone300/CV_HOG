#include "hog.h"
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void Hog::Padding(unsigned char* src, unsigned char* dsc, int width, int height, int FilterSize, int colorNum) {
    int padSize = FilterSize / 2;
    int paddedWidth = width + 2 * padSize;
    int paddedHeight = height + 2 * padSize;

    // 패딩된 이미지 초기화 (0으로 설정)
    memset(dsc, 0, paddedWidth * paddedHeight * colorNum * sizeof(unsigned char));

    // 원본 이미지를 패딩된 이미지의 중앙에 복사
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int color = 0; color < colorNum; color++) {
                dsc[colorNum * ((y + padSize) * paddedWidth + (x + padSize)) + color] = src[colorNum * (y * width + x) + color];
            }
        }
    }

    // 상단과 하단 패딩
    for (int x = 0; x < width; x++) {
        for (int color = 0; color < colorNum; color++) {
            // 상단 패딩
            for (int y = 0; y < padSize; y++) {
                dsc[colorNum * (y * paddedWidth + (x + padSize)) + color] = src[colorNum * (0 * width + x) + color];
            }
            // 하단 패딩
            for (int y = 0; y < padSize; y++) {
                dsc[colorNum * ((height + padSize + y) * paddedWidth + (x + padSize)) + color] = src[colorNum * ((height - 1) * width + x) + color];
            }
        }
    }

    // 좌우 패딩
    for (int y = 0; y < paddedHeight; y++) {
        for (int color = 0; color < colorNum; color++) {
            // 좌측 패딩
            for (int x = 0; x < padSize; x++) {
                dsc[colorNum * (y * paddedWidth + x) + color] = dsc[colorNum * (y * paddedWidth + padSize) + color];
            }
            // 우측 패딩
            for (int x = 0; x < padSize; x++) {
                dsc[colorNum * (y * paddedWidth + (width + padSize + x)) + color] = dsc[colorNum * (y * paddedWidth + (width + padSize - 1)) + color];
            }
        }
    }
}

void Hog::sobelFilter(unsigned char* src, Mat& angle, Mat& scalar, int x, int y, int width, int height, int FilterSize, int colorNum) {
    int padSize = FilterSize / 2;
    float filterX[9] = { -1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f }; // 수평 마스크
    float filterY[9] = { -1.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f };  // 수직 마스크
    double maxS = -1.0;
    double finalA = 0.0;

    for (int color = 0; color < colorNum; color++) {
        double totalX = 0.0;
        double totalY = 0.0;
        int idx = 0;
        for (int i = -padSize; i <= padSize; i++) {
            for (int j = -padSize; j <= padSize; j++) {
                int pixelVal = src[colorNum * ((y + i) * width + (x + j)) + color];
                totalX += filterX[idx] * pixelVal;
                totalY += filterY[idx] * pixelVal;
                idx++;
            }
        }
        double tempS = sqrt(totalX * totalX + totalY * totalY);
        double tempA = atan2(totalY, totalX) * (180.0 / CV_PI);
        if (tempA < 0) tempA += 180.0; // 각도를 [0,180) 범위로 조정
        if (tempS > maxS) {
            maxS = tempS;
            finalA = tempA;
        }
    }

    angle.at<double>(y - padSize, x - padSize) = finalA;
    scalar.at<double>(y - padSize, x - padSize) = maxS;
}

void Hog::oneDFilter(unsigned char* src, Mat& angle, Mat& scalar, int x, int y, int width, int height, int FilterSize, int colorNum) {
    int padSize = FilterSize / 2;
    int filterX1[3] = { -1, 0, 1 };
    int filterY1[3] = { -1, 0, 1 };
    double maxS = -1.0;
    double finalA = 0.0;

    for (int color = 0; color < colorNum; color++) {
        double totalX = 0.0;
        double totalY = 0.0;
        // X 방향 그래디언트 계산
        for (int j = -1; j <= 1; j++) {
            int pixelVal = src[colorNum * (y * width + (x + j)) + color];
            totalX += filterX1[j + 1] * pixelVal;
        }
        // Y 방향 그래디언트 계산
        for (int i = -1; i <= 1; i++) {
            int pixelVal = src[colorNum * ((y + i) * width + x) + color];
            totalY += filterY1[i + 1] * pixelVal;
        }
        double tempS = sqrt(totalX * totalX + totalY * totalY);
        double tempA = atan2(totalY, totalX) * (180.0 / CV_PI);
        if (tempA < 0) tempA += 180.0; // 각도를 [0,180) 범위로 조정
        if (tempS > maxS) {
            maxS = tempS;
            finalA = tempA;
        }
    }

    angle.at<double>(y - padSize, x - padSize) = finalA;
    scalar.at<double>(y - padSize, x - padSize) = maxS;
}

void Hog::RunSobelFilter(unsigned char* src, Mat& angle, Mat& scalar, int width, int height, int FilterSize, int colorNum, int mode) {
    int padSize = FilterSize / 2;
    int paddedWidth = width + 2 * padSize;
    int paddedHeight = height + 2 * padSize;
    Mat paddedImg(paddedHeight, paddedWidth, CV_8UC3);
    Padding(src, paddedImg.data, width, height, FilterSize, colorNum);

    // angle과 scalar 행렬 초기화
    angle = Mat::zeros(height, width, CV_64F);
    scalar = Mat::zeros(height, width, CV_64F);

    for (int y = padSize; y < paddedHeight - padSize; y++) {
        for (int x = padSize; x < paddedWidth - padSize; x++) {
            if (mode == 1)
                sobelFilter(paddedImg.data, angle, scalar, x, y, paddedWidth, paddedHeight, FilterSize, colorNum);
            else if (mode == 2)
                oneDFilter(paddedImg.data, angle, scalar, x, y, paddedWidth, paddedHeight, FilterSize, colorNum);
        }
    }
}

vector<double> Hog::getHisto(Mat& angle, Mat& scalar, int x, int y, int width, int height) {
    int size = this->cellSize;
    vector<double> angleHisto(9, 0.0); // 9개의 빈(bin)
    double a, s;
    for (int i = 0; i < size; i++) {
        if (y + i >= height) continue;
        for (int j = 0; j < size; j++) {
            if (x + j >= width) continue;
            a = angle.at<double>(y + i, x + j);
            s = scalar.at<double>(y + i, x + j);
            int bin = static_cast<int>(a / 20.0) % 9;
            double diff = (a - (bin * 20.0)) / 20.0;
            angleHisto[bin] += (1.0 - diff) * s;
            angleHisto[(bin + 1) % 9] += diff * s;
        }
    }
    return angleHisto;
}

vector<vector<double>> Hog::getAllHisto(unsigned char* src, Mat& angle, Mat& scalar, int width, int height) {
    int size = this->cellSize;
    int widthCellNum = (width + size - 1) / size; // 모든 픽셀이 포함되도록 보장
    int heightCellNum = (height + size - 1) / size;
    int cellNum = widthCellNum * heightCellNum;
    vector<vector<double>> histo(cellNum, vector<double>(9, 0)); // 9개의 빈(bin)
    int idx;
    for (int y = 0; y < height; y += size) {
        for (int x = 0; x < width; x += size) {
            idx = (y / size) * widthCellNum + (x / size);
            histo[idx] = getHisto(angle, scalar, x, y, width, height);
        }
    }

    return histo;
}

vector<double> Hog::normalize(vector<vector<double>> allHisto, int width, int height) {
    int widthCellNum = (width + this->cellSize - 1) / this->cellSize;
    int heightCellNum = (height + this->cellSize - 1) / this->cellSize;
    int blockSize = this->blockSize;
    int blockNumW = widthCellNum - blockSize + 1;
    int blockNumH = heightCellNum - blockSize + 1;

    vector<double> featureVector;

    for (int y = 0; y < blockNumH; y++) {
        for (int x = 0; x < blockNumW; x++) {
            vector<double> blockHist;

            for (int dy = 0; dy < blockSize; dy++) {
                for (int dx = 0; dx < blockSize; dx++) {
                    int cellX = x + dx;
                    int cellY = y + dy;
                    int idx = cellY * widthCellNum + cellX;

                    // 셀 히스토그램 추가
                    blockHist.insert(blockHist.end(), allHisto[idx].begin(), allHisto[idx].end());
                }
            }

            // L2-Hys 정규화
            double total = 0.0;
            for (double val : blockHist) {
                total += val * val;
            }
            double norm = sqrt(total + 1e-6);

            for (double& val : blockHist) {
                val /= norm;
                if (val > 0.2) val = 0.2; // threshold
            }

            total = 0.0;
            for (double val : blockHist) {
                total += val * val;
            }
            norm = sqrt(total + 1e-6);

            for (double& val : blockHist) {
                val /= norm;
            }

            featureVector.insert(featureVector.end(), blockHist.begin(), blockHist.end());
        }
    }

    return featureVector;
}

vector<double> Hog::getFeature(Mat t) {
    int width = t.cols;
    int height = t.rows;
    int FilterSize = 3; // 기본 필터 크기
    int colorNum = t.channels(); // 컬러 채널 수

    unsigned char* srcData = t.data;

    Mat angle, scalar;

    int mode = 1; //(1: sobelFilter, 2: oneDFilter)
    RunSobelFilter(srcData, angle, scalar, width, height, FilterSize, colorNum, mode);

    vector<vector<double>> allHisto = getAllHisto(srcData, angle, scalar, width, height);

    vector<double> featureVector = normalize(allHisto, width, height);

    return featureVector; 
}

int Hog::getCellSize() {
    return this->cellSize;
}

int Hog::getBlockSize() {
    return this->blockSize;
}

void Hog::setCellSize(int size) {
    this->cellSize = size;
}

void Hog::setBlockSize(int size) {
    this->blockSize = size;
}

int Hog::getWidth() {
    return this->hogWidth;
}

int Hog::getHeight() {
    return this->hogHeight;
}