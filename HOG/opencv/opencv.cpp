#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

#define WINDOWSIZE_WIDTH 64
#define WINDOWSIZE_HEIGHT 128
#define CELLSIZE 8
#define BLOCKSIZE 16

using namespace cv;

Mat Padding(Mat scr, int width, int height, int FilterSize, int type);
void Filtering(Mat& scr, Mat& dst, Mat& filter, int v);
void getMagPhase(const Mat& x, const Mat& y, Mat& mag, Mat& phase);
int op();
void GetHistogram(const Mat& mag, const Mat& phase, std::vector<float>& hist);
void getBlock(const Mat& mag, const Mat& phase, int x, int y, std::vector<float>& hist);
float* getCell(const Mat& mag, const Mat& phase, int x, int y);


int main() {
    Mat t = imread("lena.jpg", IMREAD_GRAYSCALE);
    Mat scr;
    resize(t, scr, Size(WINDOWSIZE_WIDTH, WINDOWSIZE_HEIGHT));
    if (scr.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    Mat sobel_x = (Mat_<float>(3, 3) << -1, -2, -1,
        0, 0, 0,
        1, 2, 1);
    Mat sobel_y = (Mat_<float>(3, 3) << -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1);

    Mat x(scr.size(), CV_32F);
    Mat y(scr.size(), CV_32F);

    Filtering(scr, x, sobel_x, 1);
    Filtering(scr, y, sobel_y, 1);

    Mat mag(scr.size(), CV_32F);
    Mat phase(scr.size(), CV_32F);

    getMagPhase(x, y, mag, phase);
    double minVal, maxVal;
    Point minLoc, maxLoc;

    // 최솟값과 최댓값 계산
    minMaxLoc(phase, &minVal, &maxVal, &minLoc, &maxLoc);
    std::cout << "최솟값: " << minVal << ", 위치: " << minLoc << std::endl;
    std::cout << "최댓값: " << maxVal << ", 위치: " << maxLoc << std::endl;
   
    //mag.convertTo(mag, CV_8UC1);
    //imshow("Magnitude", mag);

    std::vector<float> histogram;
    GetHistogram(mag, phase, histogram);
    std::cout << std::endl <<std::endl<< histogram.size() << std::endl << std::endl;

    imshow("input", t);
    imshow("scr", scr);
    
    //op();
    waitKey(0);
    return 0;
}

void GetHistogram(const Mat& mag, const Mat& phase, std::vector<float>& hist) {
    int height = mag.rows;
    int width = mag.cols;
    int cnt1 = 0;
    for (int h = 0; h < height-CELLSIZE; h += CELLSIZE){
        cnt1++;
        int cnt2 = 0;
        for (int w = 0; w < width-CELLSIZE; w += CELLSIZE) {
            getBlock(mag, phase, w, h, hist);
            cnt2++;
        }
        std::cout << std::endl << "cnt2 갯수" << cnt2 << std::endl;
    }
    std::cout << std::endl << "cnt1 갯수" << cnt1 << std::endl;
    //128 -> 105
}

void getBlock(const Mat& mag, const Mat& phase, int x, int y, std::vector<float>& hist) {

    float* tmp;
    std::vector<float> L;
    float total;
    
    for (int h = 0; h < BLOCKSIZE; h += CELLSIZE) for (int w = 0; w < BLOCKSIZE; w += CELLSIZE) {
        tmp = getCell(mag,phase,x+w, y+h);
        total = 0;
        // 정규화 과정
        for (int i = 0; i < 9; i++) {
            total = total + tmp[i]*tmp[i];
            L.push_back(tmp[i]);
        }
        total = sqrt(total);
        for (int i = 0; i < 9; i++) {
            hist.push_back(L[i] / total);
        }
        
    }
    
}

float* getCell(const Mat& mag,const Mat& phase, int x, int y) {

    float arr[9] = { 0 };
    int i1,i2;
    float di;
    float t;
    for (int h = y; h < CELLSIZE; h++) for (int w = x; w < CELLSIZE; w++) {
        t = phase.at<float>(h, w);
        i1 = t / 20;
        di = t - i1*20;
        arr[i1] += mag.at<float>(h,w) * (20 - di) / 20;
        i2 = (i1 + 1)%9;
        arr[i2] += mag.at<float>(h,w) * di / 20;
        
    }

    return arr;
}

int op() {
    Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "이미지를 로드할 수 없습니다." << std::endl;
        return -1;
    }

    // Sobel 필터를 사용하여 x 및 y 방향 그라디언트 계산
    Mat grad_x, grad_y;
    Sobel(img, grad_x, CV_32F, 1, 0, 3); // x 방향 그라디언트
    Sobel(img, grad_y, CV_32F, 0, 1, 3); // y 방향 그라디언트

    // 그라디언트 크기 계산
    Mat grad_magnitude;
    magnitude(grad_x, grad_y, grad_magnitude);

    // 그라디언트 크기 값을 8비트로 변환하여 시각화
    Mat grad_magnitude_display;
    grad_magnitude.convertTo(grad_magnitude_display, CV_8U);

    // 결과 이미지 출력
    imshow("Gradient Magnitude", grad_magnitude_display);
    return 0;
}

void getMagPhase(const Mat& x, const Mat& y, Mat& mag, Mat& phase) {
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            float gx = x.at<float>(i, j);
            float gy = y.at<float>(i, j);

            mag.at<float>(i, j) = sqrt(gx * gx + gy * gy);
            float angle = atan2(gy, gx) * 180 / CV_PI;  
            phase.at<float>(i, j) = abs(angle);  
        }
    }
}

void Filtering(Mat& scr, Mat& dst, Mat& filter, int v) {
    int height = scr.rows;
    int width = scr.cols;
    int n = 2 * v + 1;

    Mat pad = Padding(scr, width, height, n, dst.type());
    int PadSize = n / 2;

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            double conv = 0.0;
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    conv += pad.at<float>(h + j, w + i) * filter.at<float>(j, i);
                }
            }
            dst.at<float>(h, w) = static_cast<float>(conv);
        }
    }
}

Mat Padding(Mat s, int width, int height, int FilterSize, int type) {
    int PadSize = FilterSize / 2;
    int nheight = height + 2 * PadSize;
    int nwidth = width + 2 * PadSize;

    Mat scr;
    s.convertTo(scr, CV_32F);
    Mat rtn(nheight, nwidth, CV_32F, Scalar(0));

    // 원본 이미지 가운데 넣기
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            rtn.at<float>(h + PadSize, w + PadSize) = scr.at<float>(h, w);
        }
    }

    // 위쪽과 아래쪽 패딩
    for (int w = 0; w < width; w++) {
        for (int h = 0; h < PadSize; h++) {
            rtn.at<float>(h, w + PadSize) = scr.at<float>(0, w);  // 위쪽
            rtn.at<float>(h + height + PadSize, w + PadSize) = scr.at<float>(height - 1, w);  // 아래쪽
        }
    }

    // 왼쪽과 오른쪽 패딩
    for (int h = 0; h < nheight; h++) {
        for (int w = 0; w < PadSize; w++) {
            rtn.at<float>(h, w) = rtn.at<float>(h, PadSize);  // 왼쪽
            rtn.at<float>(h, w + PadSize + width) = rtn.at<float>(h, PadSize + width - 1);  // 오른쪽
        }
    }

    return rtn;
}
