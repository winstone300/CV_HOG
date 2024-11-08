#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

#define WINDOWSIZE_WIDTH 240
#define WINDOWSIZE_HEIGHT 320
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
void getBlock_Hys(const Mat& mag, const Mat& phase, int x, int y, std::vector<float>& hist);

int main() {
    Mat t = imread("people.png", IMREAD_GRAYSCALE);
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
    
    Mat hogImage = visualizeHOG(histogram, CELLSIZE, 18, WINDOWSIZE_WIDTH, WINDOWSIZE_HEIGHT);
    Mat combine;
    addWeighted(scr, 1.0, hogImage, 0.2, 0, combine);
    imshow("combine",combine);


    HOGDescriptor hog(
        Size(WINDOWSIZE_WIDTH, WINDOWSIZE_HEIGHT),   // 윈도우 사이즈
        Size(BLOCKSIZE, BLOCKSIZE),    // 블록 사이즈
        Size(CELLSIZE, CELLSIZE),      // 블록 스트라이드
        Size(CELLSIZE, CELLSIZE),      // 셀 사이즈
        18,                // 빈의 수 (각 히스토그램 방향 개수)
        1,                 // 히스토그램 계층의 수
        -1,                // 가우시안 필터 시그마 (기본값)
        HOGDescriptor::L2Hys, // 히스토그램 정규화 방법
        0.2,               // L2-Hys 정규화 클립핑 값
        true               // 부호 있는 그라디언트 사용 여부 (signedGradient)
    );
    std::vector<float> descriptors;
    hog.compute(scr, descriptors, Size(CELLSIZE, CELLSIZE));
    

    Mat hogImage2 = visualizeHOG(descriptors, CELLSIZE, 18, WINDOWSIZE_WIDTH, WINDOWSIZE_HEIGHT);
    Mat combine2;
    addWeighted(scr, 1.0, hogImage2, 0.2, 0, combine2);
    imshow("opencv", combine2);

    std::cout << std::endl << "mine, opencv\n";
    for (int i = 0; i < 50; i++) {
        std::cout << histogram[i] << ", ";
        std::cout << descriptors[i] << std::endl;
    }
    std::cout << descriptors.size() << std::endl;
    std::cout << histogram.size() << std::endl;
    

    waitKey(0);
    return 0;
}


Mat visualizeHOG(const std::vector<float>& featureVector, int cellSize, int binSize, int width, int height) {
    int numCellsX = width / cellSize;
    int numCellsY = height / cellSize;
    int angleUnit = 360 / binSize;

    // 시각화 이미지 생성
    Mat hogImage(height, width, CV_8UC1, Scalar(255));

    // 히스토그램 값의 최대값 찾기 (정규화를 위해)
    float maxVal = *std::max_element(featureVector.begin(), featureVector.end());

    int index = 0;
    for (int y = 0; y < numCellsY; y++) {
        for (int x = 0; x < numCellsX; x++) {
            // 현재 셀의 히스토그램 가져오기
            std::vector<float> hist(binSize);
            for (int i = 0; i < binSize; i++) {
                hist[i] = featureVector[index++] / maxVal;  // 정규화
            }

            // 셀의 중심 좌표
            Point cellCenter(x * cellSize + cellSize / 2, y * cellSize + cellSize / 2);

            // 히스토그램을 기반으로 선 그리기
            for (int bin = 0; bin < binSize; bin++) {
                float magnitude = hist[bin] * cellSize / 2;  // 상대적인 크기 조정
                float angle = bin * angleUnit * CV_PI / 180.0;  // 0~360도 각도 사용, 라디안 단위 변환

                // 선의 시작점과 끝점 계산 (y좌표에 주의)
                Point startPoint(
                    cellCenter.x + cos(angle) * magnitude,
                    cellCenter.y - sin(angle) * magnitude
                );
                Point endPoint(
                    cellCenter.x - cos(angle) * magnitude,
                    cellCenter.y + sin(angle) * magnitude
                );

                // 선 그리기 (Scalar(0) 사용하여 흑백으로 표시)
                line(hogImage, startPoint, endPoint, Scalar(0), 1);
            }
        }
    }

    return hogImage;
}

void GetHistogram(const Mat& mag, const Mat& phase, std::vector<float>& hist) {
    int height = mag.rows;
    int width = mag.cols;

    for (int h = 0; h < height-CELLSIZE; h += CELLSIZE){

        for (int w = 0; w < width-CELLSIZE; w += CELLSIZE) {
            getBlock_Hys(mag, phase, w, h, hist);

        }

    }

}

void getBlock_Hys(const Mat& mag, const Mat& phase, int x, int y, std::vector<float>& hist) {
    std::vector<float> blockHist;

    for (int h = 0; h < BLOCKSIZE; h += CELLSIZE) {
        for (int w = 0; w < BLOCKSIZE; w += CELLSIZE) {
            float* cellHist = getCell(mag, phase, x + w, y + h);
            blockHist.insert(blockHist.end(), cellHist, cellHist + 18);
            delete[] cellHist;
        }
    }

    // L2-Hys 정규화
    float sum = 0.0f;
    for (float val : blockHist) {
        sum += val * val;
    }
    sum = sqrt(sum + 1e-6f);  // 작은 값을 더하여 0으로 나누는 것을 방지

    for (float& val : blockHist) {
        val /= sum;
        // 임계값 적용 (예: 0.2)
        if (val > 0.2f) val = 0.2f;
    }

    // 다시 정규화
    sum = 0.0f;
    for (float val : blockHist) {
        sum += val * val;
    }
    sum = sqrt(sum + 1e-6f);

    for (float& val : blockHist) {
        val /= sum;
        hist.push_back(val);
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
        for (int i = 0; i < 18; i++) {
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

float* getCell(const Mat& mag, const Mat& phase, int x, int y) {
    float* arr = new float[18]();
    int bin1, bin2;
    float binSize = 20.0f; // 각 빈이 커버하는 각도 범위

    for (int h = y; h < y + CELLSIZE; h++) {
        for (int w = x; w < x + CELLSIZE; w++) {
            float angle = phase.at<float>(h, w);
            float magnitude = mag.at<float>(h, w);

            float bin = angle / binSize;
            bin1 = static_cast<int>(bin) % 18;
            bin2 = (bin1 + 1) % 18;
            float binFraction = bin - bin1;

            // 보간하여 히스토그램 값 누적
            arr[bin1] += magnitude * (1 - binFraction);
            arr[bin2] += magnitude * binFraction;
        }
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
            if (angle < 0) angle += 360;  // 각도를 0~360도로 조정
            phase.at<float>(i, j) = angle;
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
