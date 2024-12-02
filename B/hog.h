#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

class Hog {
public:

	//edge검출 필터(lena영상 기준 400일때 가장 성능이 좋음)
	void sobelFilter(unsigned char* src, Mat& angle, Mat& scalar, int x, int y, int width, int height, int FilterSize, int colorNum);
	void oneDFilter(unsigned char* src, Mat& angle, Mat& scalar, int x, int y, int width, int height, int FilterSize, int colorNum);
	//sobel필터 작동(dsc로 필터링 결과 출력)
	void RunSobelFilter(unsigned char* src, Mat& angle, Mat& scalar, int width, int height, int FilterSize, int colorNum,int mode);
	//패딩함수(원본이미지,출력이미지,너비,높이,필터크기(n),컬러개수)
	void Padding(unsigned char* src, unsigned char* dsc, int width, int height, int FilterSize, int colorNum);
	// src = 학습이미지, size = cell크기 , width,height = 이미지 크기
	vector<vector<double>>getAllHisto(unsigned char* src, Mat& angle, Mat& scalar, int width, int height);
	//x,y좌표를 (size x size)크기로 탐색해 histogram을 얻는것
	vector<double> getHisto(Mat& angle, Mat& scalar, int x, int y, int width,int height);
	// widthCellNum:가로 셀 개수, heightCellNum:세로 셀 개수, size: 블록 크기
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