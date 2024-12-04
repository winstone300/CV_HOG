#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

class Hog {
public:

	//edge���� ����(lena���� ���� 400�϶� ���� ������ ����)
	void sobelFilter(unsigned char* src, Mat& angle, Mat& scalar, int x, int y, int width, int height, int FilterSize, int colorNum);
	void oneDFilter(unsigned char* src, Mat& angle, Mat& scalar, int x, int y, int width, int height, int FilterSize, int colorNum);
	//sobel���� �۵�(dsc�� ���͸� ��� ���)
	void RunSobelFilter(unsigned char* src, Mat& angle, Mat& scalar, int width, int height, int FilterSize, int colorNum,int mode);
	//�е��Լ�(�����̹���,����̹���,�ʺ�,����,����ũ��(n),�÷�����)
	void Padding(unsigned char* src, unsigned char* dsc, int width, int height, int FilterSize, int colorNum);
	// src = �н��̹���, size = cellũ�� , width,height = �̹��� ũ��
	vector<vector<double>>getAllHisto(unsigned char* src, Mat& angle, Mat& scalar, int width, int height);
	//x,y��ǥ�� (size x size)ũ��� Ž���� histogram�� ��°�
	vector<double> getHisto(Mat& angle, Mat& scalar, int x, int y, int width,int height);
	// widthCellNum:���� �� ����, heightCellNum:���� �� ����, size: ��� ũ��
	vector<double> normalize(vector<vector<double>> allHisto, int width, int height);
	vector<double> getFeature(Mat t);
	int getCellSize();
	int getBlockSize();
	int getWidth();
	int getHeight();
	void setCellSize(int size);
	void setBlockSize(int size);
private:
	int cellSize = 8;
	int blockSize = 2;
	int hogWidth = 64;
	int hogHeight = 128;
};