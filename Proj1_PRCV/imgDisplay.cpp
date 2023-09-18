#include<opencv2/opencv.hpp>
#include<string>
#include<iostream>
using namespace std;
using namespace cv;
int main(int argc, char* argv[]) {
    Mat img = imread("abc.png");
    namedWindow("image", WINDOW_NORMAL);
    imshow("image", img);
    waitKey(10);
    while (true)
    {
        char c;
        cin >> c;
        waitKey(10);
        if (c == 'q')
        {
            destroyAllWindows();
            break;
        }
    }
    return 0;
}