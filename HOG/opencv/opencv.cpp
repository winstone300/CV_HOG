#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

#define WINDOWSIZE_WIDTH 512
#define WINDOWSIZE_HEIGHT 512
#define CELLSIZE 16
#define BLOCKSIZE 32

using namespace cv;

Mat Padding(Mat scr, int width, int height, int FilterSize, int type);
void Filtering(Mat& scr, Mat& dst, Mat& filter, int v);
void getMagPhase(const Mat& x, const Mat& y, Mat& mag, Mat& phase);
int op();
void GetHistogram(const Mat& mag, const Mat& phase, std::vector<float>& hist);
void getBlock(const Mat& mag, const Mat& phase, int x, int y, std::vector<float>& hist);
float* getCell(const Mat& mag, const Mat& phase, int x, int y);
Mat visualizeHOG(const std::vector<float>& featureVector, int cellSize, int binSize, int width, int height);

int main() {
    Mat t = imread("lena.jpg", IMREAD_GRAYSCALE);
    Mat scr = t.clone();
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

   
    //mag.convertTo(mag, CV_8UC1);
    //imshow("Magnitude", mag);

    std::vector<float> histogram;
    GetHistogram(mag, phase, histogram);
    std::cout << std::endl <<std::endl<< histogram.size() << std::endl << std::endl;



    //imshow("input", t);
    //imshow("scr", scr);
    
    Mat hogImage = visualizeHOG(histogram, CELLSIZE, 9, WINDOWSIZE_WIDTH, WINDOWSIZE_HEIGHT);
    Mat combine;
    addWeighted(scr, 1.0, hogImage, 0.2, 0, combine);
    imshow("combine",combine);


    HOGDescriptor hog(
        Size(WINDOWSIZE_WIDTH, WINDOWSIZE_HEIGHT),   // 윈도우 사이즈
        Size(BLOCKSIZE, BLOCKSIZE),    // 블록 사이즈
        Size(CELLSIZE, CELLSIZE),      // 블록 스트라이드
        Size(CELLSIZE, CELLSIZE),      // 셀 사이즈
        9                // 빈의 수 (각 히스토그램 방향 개수)
    );
    std::vector<float> descriptors;
    hog.compute(scr, descriptors, Size(CELLSIZE, CELLSIZE));
    

    Mat hogImage2 = visualizeHOG(descriptors, CELLSIZE, 9, WINDOWSIZE_WIDTH, WINDOWSIZE_HEIGHT);
    Mat combine2;
    addWeighted(scr, 1.0, hogImage2, 0.2, 0, combine2);
    imshow("opencv", combine2);

    std::cout << std::endl << "mine, opencv\n";
    for (int i = 0; i < 50; i++) {
        std::cout << histogram[i] << ", ";
        std::cout << descriptors[i] << std::endl;
    }


    waitKey(0);
    return 0;
}




// HOG 특징 벡터를 시각화하는 함수
Mat visualizeHOG(const std::vector<float>& featureVector, int cellSize, int binSize, int width, int height) {
    int numCellsX = width / cellSize;
    int numCellsY = height / cellSize;
    int angleUnit = 360 / binSize;

    // 시각화 이미지 생성
    Mat hogImage(height, width, CV_8UC1, Scalar(255, 255, 255));

    int index = 0;
    for (int y = 0; y < numCellsY; y++) {
        for (int x = 0; x < numCellsX; x++) {
            // 현재 셀의 히스토그램 가져오기
            std::vector<float> hist(binSize);
            for (int i = 0; i < binSize; i++) {
                hist[i] = featureVector[index++];
            }

            // 셀의 중심 좌표
            Point cellCenter(x * cellSize + cellSize / 2, y * cellSize + cellSize / 2);

            // 히스토그램을 기반으로 선 그리기
            for (int bin = 0; bin < binSize; bin++) {
                float magnitude = hist[bin];
                float angle = bin * angleUnit * CV_PI / 180.0; // 라디안 단위로 변환

                // 선의 시작점과 끝점 계산
                Point startPoint(cellCenter.x + cos(angle) * magnitude * cellSize / 2,
                    cellCenter.y + sin(angle) * magnitude * cellSize / 2);
                Point endPoint(cellCenter.x - cos(angle) * magnitude * cellSize / 2,
                    cellCenter.y - sin(angle) * magnitude * cellSize / 2);

                // 선 그리기
                line(hogImage, startPoint, endPoint, Scalar(0, 0, 255), 1);
            }
        }
    }

    // 시각화된 HOG 이미지 출력
    //imshow("HOG Visualization", hogImage);

    return hogImage;
}


void GetHistogram(const Mat& mag, const Mat& phase, std::vector<float>& hist) {
    int height = mag.rows;
    int width = mag.cols;

    for (int h = 0; h < height-CELLSIZE; h += CELLSIZE){

        for (int w = 0; w < width-CELLSIZE; w += CELLSIZE) {
            getBlock(mag, phase, w, h, hist);

        }

    }

}

void getBlock(const Mat& mag, const Mat& phase, int x, int y, std::vector<float>& hist) {

    float* tmp;
    std::vector<float> L;
    float total;
    total = 0;
    for (int h = 0; h < BLOCKSIZE; h += CELLSIZE) for (int w = 0; w < BLOCKSIZE; w += CELLSIZE) {
        tmp = getCell(mag, phase, x + w, y + h);
        
        // 정규화 과정
        for (int i = 0; i < 9; i++) {
            total = total + tmp[i] * tmp[i];
            L.push_back(tmp[i]);
        }
        delete[] tmp;
    }
    total = sqrt(total);
    for (int i = 0; i < L.size(); i++) {
        hist.push_back(L[i] / total);
    }
    
    
    
}

float* getCell(const Mat& mag,const Mat& phase, int x, int y) {

    float* arr = new float[9]();
    int i1,i2;
    float di;
    float t;
    for (int h = y; h < y+CELLSIZE; h++) for (int w = x; w < x+CELLSIZE; w++) {
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
