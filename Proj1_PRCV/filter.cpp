#include<opencv2/opencv.hpp>
#include<string>
#include<iostream>

using namespace std;
using namespace cv;
int reflect(int M, int x)
{
    if (x < 0)
    {
        return -x - 1;
    }
    if (x >= M)
    {
        return 2 * M - x - 1;
    }
    return x;
}

int blur5x5(const cv::Mat& src, cv::Mat& dst)
{
    Mat tmp;
    float sum, x1, y1;
    double coeffs[] = { 0.0545, 0.2442, 0.4026, 0.2442, 0.0545 };
    dst = src.clone();
    tmp = src.clone();

    // along y - direction
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            sum = 0.0;
            for (int i = -2; i <= 2; i++) {
                y1 = reflect(src.rows, y - i);
                sum = sum + coeffs[i + 2] * src.at<uchar>(y1, x);
            }
            tmp.at<uchar>(y, x) = sum;
        }
    }

    // along x - direction
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            sum = 0.0;
            for (int i = -2; i <= 2; i++) {
                x1 = reflect(src.cols, x - i);
                sum = sum + coeffs[i + 2] * tmp.at<uchar>(y, x1);
            }
            dst.at<uchar>(y, x) = sum;
        }
    }

    namedWindow("after");
    imshow("after", dst);

    namedWindow("before");
    imshow("before", src);

    waitKey();
    return 0;
}

int greyscale(cv::Mat& src, cv::Mat& dst) {
    cv::cvtColor(src, dst, COLOR_BGR2HSV);
    cv::Mat hsv_channels[3];
    cv::split(dst, hsv_channels);
    cv::imshow("altGreyVideo", hsv_channels[2]);
    waitKey(0);
    return 0;
}
cv::Vec3s applyFilterS(const cv::Mat& src, const cv::Mat& filter, int sRow, int sCol)
{
    cv::Vec3i result = { 0, 0, 0 };
    float coePos = 0.0, coeNeg = 0.0;
    int fHalfRow = filter.rows / 2;
    int fHalfCol = filter.cols / 2;
    int row, col, cha;
    for (row = 0; row < filter.rows; row++)
    {
        for (col = 0; col < filter.cols; col++)
        {
            float filterVal = filter.at<float>(row, col);
            if (filterVal > 0)
                coePos += filterVal;
            else
                coeNeg -= filterVal;
            for (cha = 0; cha < 3; cha++)
                result[cha] += filterVal * src.at<cv::Vec3b>(sRow + row - fHalfRow, sCol + col - fHalfCol)[cha];
        }
    }
    coePos = coePos > coeNeg ? coePos : coeNeg;
    cv::Vec3s ret;
    for (cha = 0; cha < 3; cha++)
        ret[cha] = (short)(result[cha] / coePos);
    return ret;
}
int sobelX3x3(const cv::Mat& src, cv::Mat& dst)
{
    dst.create(src.size(), CV_16SC3);
    float dummy_col_data[3][1] = { {1}, {2}, {1} };
    cv::Mat cof = cv::Mat(3, 1, CV_32F, dummy_col_data);
    float dummy_row_data[3] = { -1, 0, 1 };
    cv::Mat rof = cv::Mat(1, 3, CV_32F, dummy_row_data);
    int row, col;
    cv::Vec3s* result;
    for (row = 1; row < src.rows - 1; row++)
    {
        result = dst.ptr<cv::Vec3s>(row);
        for (col = 1; col < src.cols - 1; col++)
            result[col] = applyFilterS(src, cof, row, col);
    }
    for (row = 1; row < src.rows - 1; row++)
    {
        result = dst.ptr<cv::Vec3s>(row);
        for (col = 1; col < src.cols - 1; col++)
            result[col] = applyFilterS(src, rof, row, col);
    }
    //dst.convertTo(dst, CV_8UC3);
    //imshow("Sobel X", dst);

    //waitKey();

    return 0;
}
int sobelY3x3(const cv::Mat& src, cv::Mat& dst)
{
    dst.create(src.size(), CV_16SC3);
    float dummy_col_data[3][1] = { {-1}, {0}, {1} };
    cv::Mat cof = cv::Mat(3, 1, CV_32F, dummy_col_data);
    float dummy_row_data[3] = { 1, 2, 1 };
    cv::Mat rof = cv::Mat(1, 3, CV_32F, dummy_row_data);
    int row, col;
    cv::Vec3s* result;
    for (row = 1; row < src.rows - 1; row++)
    {
        result = dst.ptr<cv::Vec3s>(row);
        for (col = 1; col < src.cols - 1; col++)
            result[col] = applyFilterS(src, rof, row, col);
    }
    for (row = 1; row < src.rows - 1; row++)
    {
        result = dst.ptr<cv::Vec3s>(row);
        for (col = 1; col < src.cols - 1; col++)
            result[col] = applyFilterS(src, cof, row, col);
    }
    //dst.convertTo(dst, CV_8UC3);
    //imshow("Sobel Y", dst);

    //waitKey();

    return 0;
}
int magnitude(const cv::Mat& sx, const cv::Mat& sy, cv::Mat& dst)
{
    dst.create(sx.size(), CV_8UC3);
    int row, col;
    short cha, x, y;
    for (row = 0; row < sx.rows; row++)
    {
        const cv::Vec3s* xpt = sx.ptr<cv::Vec3s>(row);
        const cv::Vec3s* ypt = sy.ptr<cv::Vec3s>(row);
        cv::Vec3b* result = dst.ptr<cv::Vec3b>(row);
        for (col = 0; col < sx.cols; col++)
        {
            for (cha = 0; cha < 3; cha++)
            {
                x = xpt[col][cha];
                y = ypt[col][cha];

                result[col][cha] = (unsigned char)
                    std::max(std::min(std::sqrt(x * x + y * y), 255.0), 0.0);
            }
        }
    }
    //dst.convertTo(dst, CV_8UC3);
    //imshow("Magnitude", dst);

    //waitKey();
    return 0;
}
int blurQuantize(const cv::Mat src, cv::Mat dst, int levels)
{
    if (blur5x5(src, dst) != 0)
        exit(-1);
    levels = 15;
    short b = 255 / levels;
    int row, col, cha;
    for (row = 0; row < src.rows; row++)
    {
        for (col = 0; col < src.cols; col++)
        {
            cv::Vec3b drr = dst.at<cv::Vec3b>(row, col);
            for (cha = 0; cha < 3; cha++)
                drr[cha] = drr[cha] / b * b;
        }
    }
    //dst.convertTo(dst, CV_8UC3);
    //imshow("blurQuantize", dst);

    //waitKey();
    return 0;
}
int cartoon(const cv::Mat& src, cv::Mat& dst, int levels, int magThreshold)
{
    cv::Mat sx, sy, gm;
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);
    magnitude(sx, sy, gm);

    blurQuantize(src, dst, levels);

    bool flag;
    int row, col, cha;
    for (row = 0; row < src.rows; row++)
    {
        cv::Vec3b* gpt = gm.ptr<cv::Vec3b>(row);
        cv::Vec3b* dpt = dst.ptr<cv::Vec3b>(row);

        for (col = 0; col < src.cols; col++)
        {
            flag = false;
            for (cha = 0; cha < 3; cha++)
            {
                if (gpt[col][cha] > magThreshold)
                {
                    flag = true;
                    break;
                }
            }
            for (cha = 0; cha < 3; cha++)
            {
                // edge
                if (flag)
                {
                    dpt[col][cha] = 0;
                }
            }
        }
    }

    return 0;
}


void addSparkles(const cv::Mat& src, cv::Mat& dst, int threshold)
{
    cv::Mat gray, sobelX, sobelY, mag;

    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Sobel(gray, sobelX, CV_16S, 1, 0);
    cv::Sobel(gray, sobelY, CV_16S, 0, 1);
    cv::convertScaleAbs(sobelX, sobelX);
    cv::convertScaleAbs(sobelY, sobelY);
    cv::addWeighted(sobelX, 0.5, sobelY, 0.5, 0, mag);

    dst = src.clone();

    for (int row = 0; row < src.rows; row++)
    {
        for (int col = 0; col < src.cols; col++)
        {
            uchar magVal = mag.at<uchar>(row, col);
            cv::Vec3b& dstVal = dst.at<cv::Vec3b>(row, col);

            if (magVal > threshold)
            {
                dstVal = cv::Vec3b(255, 255, 255);
            }
        }
    }
}


void pixelateFilter(const cv::Mat& src, cv::Mat& dst, int blockSize)
{
    dst = src.clone();
    for (int row = 0; row < src.rows; row += blockSize)
    {
        for (int col = 0; col < src.cols; col += blockSize)
        {
            int blockRows = std::min(blockSize, src.rows - row);
            int blockCols = std::min(blockSize, src.cols - col);
            cv::Vec3d sum(0, 0, 0);
            int numPixels = 0;
            for (int r = 0; r < blockRows; ++r)
            {
                for (int c = 0; c < blockCols; ++c)
                {
                    sum += src.at<cv::Vec3b>(row + r, col + c);
                    numPixels++;
                }
            }
            cv::Vec3b averageColor = sum / numPixels;
            for (int r = 0; r < blockRows; ++r)
            {
                for (int c = 0; c < blockCols; ++c)
                {
                    dst.at<cv::Vec3b>(row + r, col + c) = averageColor;
                }
            }
        }
    }
}

int main(int argc, char argv[]) {
    Mat src = cv::imread("abc.png");
    cv::Mat dst;
    char c;
    cin >> c;
    if (c == 'h')
    {
        greyscale(src, dst);
    }
    if (c == 'b')
    {
        blur5x5(src, dst);
    }
    if (c == 'x')
    {
        sobelX3x3(src, dst);
        dst.convertTo(dst, CV_8UC3);
        imshow("Sobel X", dst);
    }
    if (c == 'y')
    {
        sobelY3x3(src, dst);
        dst.convertTo(dst, CV_8UC3);
        imshow("Sobel Y", dst);
    }
    if (c == 'm')
    {
        cv::Mat sobelX;
        cv::Mat sobelY;
        sobelX3x3(src, sobelX);
        sobelY3x3(src, sobelY);
        magnitude(sobelX, sobelY, dst);
        dst.convertTo(dst, CV_8UC3);
        imshow("Magnitude", dst);
    }
    if (c == 'l')
    {
        blurQuantize(src, dst, 15);
        dst.convertTo(dst, CV_8UC3);
        imshow("Quantized", dst);
    }
    if (c == 'c')
    {
        cv::VideoCapture* capdev;
        capdev = new cv::VideoCapture(0);
        cv::namedWindow("Video", 1); // identifies a window
        cv::Mat frame;

        for (;;) {
            *capdev >> frame; // get a new frame from the camera, treat as a stream
            if (frame.empty()) {
                printf("frame is empty\n");
                break;
            }
            cartoon(frame, frame, 15, 15);
            cv::imshow("Video", frame);
            char key = cv::waitKey(1);
        }
    }
    if (c == 's')
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            std::cerr << "Unable to open the camera" << std::endl;
            return -1;
        }
        cv::namedWindow("Live Video with Sparkles", cv::WINDOW_AUTOSIZE);
        cv::Mat frame, sparkledFrame;

        while (true)
        {
            cap >> frame;
            if (frame.empty())
            {
                std::cerr << "Failed to capture a frame" << std::endl;
                break;
            }
            addSparkles(frame, sparkledFrame, 100);
            cv::imshow("Live Video with Sparkles", sparkledFrame);
            if (cv::waitKey(30) >= 0)
            {
                break;
            }
        }
        return 0;
    }

    if (c == 'p') {
        cv::VideoCapture cap(0);

        if (!cap.isOpened())
        {
            std::cerr << "Unable to open the camera" << std::endl;
            return -1;
        }

        cv::namedWindow("Live Video with Pixelated Filter", cv::WINDOW_AUTOSIZE);

        cv::Mat frame, filteredFrame;

        while (true)
        {
            cap >> frame;

            if (frame.empty())
            {
                std::cerr << "Failed to capture a frame" << std::endl;
                break;
            }

            pixelateFilter(frame, filteredFrame, 10);

            cv::imshow("Live Video with Pixelated Filter", filteredFrame);

            if (cv::waitKey(30) >= 0)
            {
                break;
            }
        }

        return 0;
    }
}
