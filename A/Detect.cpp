#include "HOG.hpp"



int main(void){
    // 예측
    HOG hog;
    Ptr<cv::ml::SVM> svm = Algorithm::load<cv::ml::SVM>("trained_svm_model2.xml");

    string path = "./TestData/test/detect";
    for (const auto& entry : fs::directory_iterator(path)) {
        
        double scaleFactor = 2; 
        double currentScale = 1.0/2.0;
        int minSizeWidth = hog.getWidth(); 
        int minSizeHeight = hog.getHeight();
        Mat img =imread((string)entry.path());
        Mat testImage = imread((string)entry.path(), IMREAD_GRAYSCALE);

        if(testImage.empty()){
            cout<<"Can't Open file"<<endl;
            continue;
        }

        int height = testImage.rows;
        int width = testImage.cols;
        
        vector<Rect> detectedRects;
        HOG hog;
        Mat resizeImage;
        resizeImage = testImage.clone();
        resize(resizeImage, resizeImage, Size(), 2.0, 2.0);
        while(1){
            
            // resize(resizeImage, resizeImage, Size(), 1.0 / scaleFactor, 1.0 / scaleFactor);

            Mat dst;
            GaussianPyramid(resizeImage, dst);            
            resizeImage = dst.clone();

            height = resizeImage.rows;
            width = resizeImage.cols;
            currentScale *= scaleFactor;
            
            if(!(minSizeWidth <= resizeImage.cols && minSizeHeight<=resizeImage.rows))break;

            for (int h = 0; h < height-hog.getHeight()-1; h += 8){

                for (int w = 0; w < width-hog.getWidth()-1; w += 8) {

                    Rect range(w,h, hog.getWidth(), hog.getHeight());
                    Mat scr = resizeImage(range);
                    // imshow("matching", scr);
                    // waitKey(0);
                    vector<float> descriptors = hog.getFeature(scr);
                    cv::Mat testFeatures = cv::Mat(descriptors).clone().reshape(1, 1);
                    float response = svm->predict(testFeatures);
                    if (response == 1) {
                        // cout<<"detect"<<endl;
                        int originalX = static_cast<int>(w * currentScale);
                        int originalY = static_cast<int>(h * currentScale);
                        int originalW = static_cast<int>(hog.getWidth() * currentScale);
                        int originalH = static_cast<int>(hog.getHeight() * currentScale);
                        // imshow("detected!!!!!11", testImage(Rect(originalX, originalY, originalW, originalH)));
                        // waitKey(0);
                        // rectangle(testImage, Rect(originalX, originalY, originalW, originalH), Scalar(0, 255, 0), 2);
                        detectedRects.push_back(Rect(originalX, originalY, originalW, originalH));
                    }
                }

            }

        }
        vector<int> weights;
        groupRectangles(detectedRects, weights, 5, 0.5);

        // 최종 검출 결과 표시
        for (const auto& rect : detectedRects) {
            rectangle(img, rect, Scalar(0, 255, 0), 2);
        }
        imwrite("./output_more/" + (string)entry.path().filename(), img);
        // imshow("Detected",img);
        // waitKey(0);
    }

    return 0;
}
