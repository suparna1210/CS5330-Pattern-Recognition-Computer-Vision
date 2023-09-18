#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main() {

    // Create a VideoCapture object and use camera to capture the video
    VideoCapture cap(0);

    // Check if camera opened successfully
    if (!cap.isOpened()) {
        cout << "Error opening video stream" << endl;
        return -1;
    }

    // Default resolutions of the frame are obtained.The default resolutions are system dependent.
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    char key;
    int i = 1;
    while (1) {

        Mat frame;

        // Capture frame-by-frame
        cap >> frame;
        // If the frame is empty, break immediately
        if (frame.empty())
            break;


        // Display the resulting frame    
        imshow("Frame", frame);
        //printf("Expected size");
        cin >> key;

        // Press  q on keyboard to  exit
        waitKey(1);
        if (key == 'q')
            break;
        // Press  s on keyboard to  save to directory
        if (key == 's') {
            stringstream ss;
            string temp;

            ss << i << ("savedimg.jpg");
            temp = ss.str();
            imwrite(temp, frame);
        }
        // Press  g on keyboard to display greyscale
        if (key == 'g')
        {
            cv::namedWindow("Greyvideo", 1);
            cv::Mat greyFrame;
            cvtColor(frame, greyFrame, COLOR_BGR2GRAY);
            cv::imshow("Greyvideo", greyFrame);
        }
        // Press  h on keyboard to display alternate greyscale
        if (key == 'h')
        {
            cv::Mat dst;
            cv::cvtColor(frame, dst, COLOR_BGR2HSV);
            cv::Mat hsv_channels[3];
            cv::split(dst, hsv_channels);
            cv::imshow("altGreyVideo", hsv_channels[2]);
        }
        i += 1;
    }

    // When everything done, release the video capture
    cap.release();


    // Closes all the frames
    return 0;
}
