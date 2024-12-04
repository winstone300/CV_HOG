#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>
// #include "tinyxml2.h"
#include <filesystem>
#include <algorithm>
#include <random>

namespace fs = std::__fs::filesystem;
using namespace std;
// using namespace tinyxml2;
using namespace cv;

class HOG
{
private:
    int WINDOWSIZE_WIDTH;
    int WINDOWSIZE_HEIGHT;
    int CELLSIZE;
    int BLOCKSIZE;

    Mat Padding(Mat scr, int width, int height, int FilterSize, int type);
    void Filtering(Mat& scr, Mat& dst, Mat& filter, int v);
    void getMagPhase(const Mat& x, const Mat& y, Mat& mag, Mat& phase);
    void GetHistogram(const Mat& mag, const Mat& phase, std::vector<float>& hist);
    void getBlock(const Mat& mag, const Mat& phase, int x, int y, std::vector<float>& hist);
    float* getCell(const Mat& mag, const Mat& phase, int x, int y);
    void getBlock_Hys(const Mat& mag, const Mat& phase, int x, int y, std::vector<float>& hist);

public:
    HOG(){
        WINDOWSIZE_WIDTH =64;
        WINDOWSIZE_HEIGHT =128;
        CELLSIZE =8;
        BLOCKSIZE =16;


    }
    HOG(int width, int height, int cell, int block){
        WINDOWSIZE_WIDTH = width;
        WINDOWSIZE_HEIGHT = height;
        CELLSIZE= cell;
        BLOCKSIZE = block;


    }

    std::vector<float> getFeature(Mat t);
    void visualizeHOG(const std::vector<float>& featureVector);

    int getHeight(){return WINDOWSIZE_HEIGHT;}
    int getWidth(){
        return WINDOWSIZE_WIDTH;
    }
    int getCellSize(){
        return CELLSIZE;
    }

};
// vector<Rect> parseXML(const string& xmlPath);
// void getData(const string& imgPath, const string& xmlPath, Mat& trainingData, Mat& labels);
void Filtering(Mat& scr, Mat& dst, Mat& filter, int v);
Mat Padding(Mat scr, int width, int height, int FilterSize, int type);
void GaussianPyramid(const Mat& src, Mat& dst);