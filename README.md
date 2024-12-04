## HOG와 SVM을 이용한 Object Detection

### HOG 구현 방식
|Folder (Branch)|Gradient 범위|정규화 방식|중복 처리|특징벡터 크기|
|:---:|:---:|:---:|:---:|:---:|
|A (jaemin)|0~360|L2-Hys|O|7560|
|B (jsh)|0~180|L2-Hys|O|3780|
|C (junho)|0~180|L2-norm|X|1152|

### 참조
+ HOG : https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
+ PETA dataset : https://mmlab.ie.cuhk.edu.hk/projects/PETA.html
