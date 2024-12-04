#include "HOG.hpp"

bool endsWith(std::string const &str, std::string const &suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}


int main(void){
    HOG hog;
    
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);               // 분류 문제 설정
    svm->setKernel(cv::ml::SVM::LINEAR);             // 선형 커널 사용
    svm->setC(1.0);                                  // 정규화 파라미터 C
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-6)); // 종료 조건

    Mat trainingData; // 모든 HOG 특징 벡터를 결합한 학습 데이터 행렬
    Mat labels;       // 각 이미지의 라벨 (1 또는 -1 등)
    string path = "./TestData/true";
    
    for (const auto& entry : fs::directory_iterator(path)) {        
        Mat roi = imread((string)entry.path());
        std::vector<float> descriptors = hog.getFeature(roi); 
        cv::Mat hogFeatures = cv::Mat(descriptors).clone().reshape(1, 1); // 특징 벡터를 행렬로 변환
        trainingData.push_back(hogFeatures); // 학습 데이터에 추가
        labels.push_back(1);
        
    }
    path = "./TestData/archive";
    for (const auto& entry : fs::directory_iterator(path)) {        
        Mat roi = imread((string)entry.path());
        if(roi.empty()) continue;
        std::vector<float> descriptors = hog.getFeature(roi); 
        cv::Mat hogFeatures = cv::Mat(descriptors).clone().reshape(1, 1); // 특징 벡터를 행렬로 변환
        trainingData.push_back(hogFeatures); // 학습 데이터에 추가
        labels.push_back(1);
        
    }
    cout<<"30%"<<endl;


    cout<<"60%"<<endl;
    path = "./TestData/false";
    
    for (const auto& entry : fs::directory_iterator(path)) {

        Mat roi = imread((string)entry.path());
        

        std::vector<float> descriptors = hog.getFeature(roi); 
        cv::Mat hogFeatures = cv::Mat(descriptors).clone().reshape(1, 1); // 특징 벡터를 행렬로 변환
        trainingData.push_back(hogFeatures); // 학습 데이터에 추가
        labels.push_back(-1);
    }

    path = "./TestData/temp";
    
    for (const auto& entry : fs::directory_iterator(path)) {

        Mat roi = imread((string)entry.path());
        
        if(roi.empty()) continue;
        std::vector<float> descriptors = hog.getFeature(roi); 
        cv::Mat hogFeatures = cv::Mat(descriptors).clone().reshape(1, 1); // 특징 벡터를 행렬로 변환
        trainingData.push_back(hogFeatures); // 학습 데이터에 추가
        labels.push_back(-1);
    }

    cout<<"Error"<<endl;

    vector<int> indices(trainingData.rows);
    iota(indices.begin(), indices.end(), 0);

    // 인덱스를 무작위로 섞기
    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    // 새로운 행렬로 재배열
    Mat shuffledTrainingData, shuffledLabels;
    for (int i : indices) {
        shuffledTrainingData.push_back(trainingData.row(i));
        shuffledLabels.push_back(labels.row(i));
    }

    trainingData = shuffledTrainingData;
    labels = shuffledLabels;

    cout<<trainingData.size()<<endl;

    cout<<"training start"<<endl;
    svm->train(trainingData, cv::ml::ROW_SAMPLE, labels); // SVM 학습
    cout<<"traing end"<<endl;


    // True Test data
    path = "./TestData/test/trueTest";
    int correct = 0;
    int ncorrect =0;
    int cnt =0;
    for (const auto& entry : fs::directory_iterator(path)) {
        // 일반 파일인 경우만
        Mat testImage = imread((string)entry.path());
        if(testImage.empty()){
            cout<<"can't open testImage True"<<endl;
            continue;
        }

        std::vector<float> descriptors = hog.getFeature(testImage); 

        cv::Mat testFeatures = cv::Mat(descriptors).clone().reshape(1, 1);
        float response = svm->predict(testFeatures);
        if (response == 1) {
            //std::cout << "Positive class" << std::endl;
            correct++;
        } else {
            //std::cout << "Negative class" << std::endl;
            ncorrect++;
            // std::cout << "False Negative" << std::endl;
            // cout<<entry.path()<<endl;
        }
        cnt++;
        
    }
    cout<<cnt<<endl;
    cout<<"TP"<<correct<<endl;
    cout<<"FN"<<ncorrect<<endl;


    //False Test Data
    path = "./TestData/test/falseTest";
    correct = 0;
    ncorrect = 0;
    cnt =0;
    for (const auto& entry : fs::directory_iterator(path)) {

        // if (entry.is_regular_file()) { // 일반 파일인 경우만
        //     Mat testImage = imread((string)entry.path());

        //     std::vector<float> descriptors = hog.getFeature(testImage); 

        //     cv::Mat testFeatures = cv::Mat(descriptors).clone().reshape(1, 1);
        //     float response = svm->predict(testFeatures);
        //     if (response == 1) {
        //         //std::cout << "Positive class" << std::endl;
        //         ncorrect++;               
        //     } else {
        //         //std::cout << "Negative class" << std::endl;
        //         correct++;
        //     }
        //     cnt++;
        // }
        Mat testImage = imread((string)entry.path());
        if(testImage.empty()){
            cout<<"can't open testImage True"<<endl;
            continue;
        }

        std::vector<float> descriptors = hog.getFeature(testImage); 

        cv::Mat testFeatures = cv::Mat(descriptors).clone().reshape(1, 1);
        float response = svm->predict(testFeatures);
        if (response == 1) {
            // std::cout << "False Positive" << std::endl;
            // cout<<entry.path()<<endl;
            ncorrect++;               
        } else {
            //std::cout << "Negative class" << std::endl;
            correct++;
        }
        cnt++;
    }
    cout<<cnt<<endl;
    cout<<"TN"<<correct<<endl;
    cout<<"FP"<<ncorrect<<endl;

    //svm->save("trained_svm_model.xml");
    

    return 0;
}


