#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#pragma warning(suppress : 4996)

using namespace cv;
using namespace std;

// 2. Preprocess input frame
Mat preprocessFrame(const Mat& inputFrame) {
    Mat processedFrame;
    GaussianBlur(inputFrame, processedFrame, Size(5, 5), 0, 0);
    cvtColor(processedFrame, processedFrame, COLOR_BGR2GRAY);

    return processedFrame;
}

//Part 1: Thresholding
Mat thresholdFrame(const Mat& processedFrame) {
    Mat thresholdedFrame;
    cv::threshold(processedFrame, thresholdedFrame, 80, 255, 1);
    return thresholdedFrame;
}


// Part 2: Morphological filtering
int morph_cleanup(cv::Mat& input, cv::Mat& output, char* morph_type) {

    typedef cv::Vec<uchar, 1> GrayVec;
    output = cv::Mat::zeros(input.size(), CV_8UC1);

    for (int row = 0; row < input.rows; row++) {

        GrayVec* input_row = input.ptr<GrayVec>(row);
        GrayVec* output_row = output.ptr<GrayVec>(row);

        for (int col = 0; col < input.cols; col++) {

            int pixel_value = input_row[col][0];

            if (strcmp(morph_type, "dilation") == 0) {
                if (pixel_value < 15) {
                    output_row[col][0] = 255;
                }
                else {
                    output_row[col][0] = 0;
                }
            }
            else if (strcmp(morph_type, "erosion") == 0) {
                if (pixel_value < 15) {
                    output_row[col][0] = 0;
                }
                else {
                    output_row[col][0] = 255;
                }
            }
        }
    }
    return 0;
}

//Part 3: Segmentation
std::tuple<Mat, Mat, Mat, Mat> Segmentation(cv::Mat& img) {
    Mat labels;
    Mat stats;
    Mat centroids;
    int num_components = cv::connectedComponentsWithStats(img, labels, stats, centroids);

    // Create a random color map for visualization
    std::vector<cv::Vec3b> colors(num_components);
    colors[0] = cv::Vec3b(0, 0, 0);  // Background color
    for (int i = 1; i < num_components; i++) {
        colors[i] = cv::Vec3b(255, 0, 0);
    }

    // Display the components that meet the minimum size requirement
    cv::Mat segmentedFrame(img.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int label = labels.at<int>(i, j);
            if (label != 0 && stats.at<int>(label, cv::CC_STAT_AREA) >= 1000) {
                segmentedFrame.at<cv::Vec3b>(i, j) = colors[label];
            }
        }
    }
    return std::make_tuple(segmentedFrame, labels, stats, centroids);
}


//Part4: compute features
std::vector<double> computeFeatures(cv::Mat& frame, cv::Mat& region_map, int region_id) {
    std::vector<double> featureVector;

    // Create a mask for the specified region
    cv::Mat label_mask = (region_map == region_id);

    // Compute the moments of the region
    cv::Moments moments = cv::moments(label_mask, true);

    // Calculate the x-axis and y-axis centroids
    double x_bar = moments.m10 / moments.m00;
    double y_bar = moments.m01 / moments.m00;
    double theta = 0.5 * atan2(2 * moments.mu11, moments.mu20 - moments.mu02);

    cv::Point2f center(x_bar, y_bar);
    cv::Point2f endpoint1(center.x + std::cos(theta) * 100, center.y + std::sin(theta) * 100);
    cv::Point2f endpoint2(center.x - std::cos(theta) * 100, center.y - std::sin(theta) * 100);

    // Calculate the oriented bounding box
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(label_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::RotatedRect box = cv::minAreaRect(contours[0]);

    cv::Point2f vertices[4];
    box.points(vertices);

    // Draw the axis of least central moment and the oriented bounding box on the frame
    cv::line(frame, endpoint1, endpoint2, cv::Scalar(0, 255, 0), 2);
    for (int i = 0; i < 4; i++) {
        cv::line(frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
    }

    // Add features to the feature vector
    featureVector.push_back(x_bar);
    featureVector.push_back(y_bar);
    featureVector.push_back(theta);
    featureVector.push_back(box.size.width);
    featureVector.push_back(box.size.height);
    featureVector.push_back(box.angle);

    return featureVector;
}


/*
  Given a filename, and image filename, and the image features, by
  default the function will append a line of data to the CSV format
  file.  If reset_file is true, then it will open the file in 'write'
  mode and clear the existing contents.

  The image filename is written to the first position in the row of
  data. The values in image_data are all written to the file as
  floats.

  The function returns a non-zero value in case of an error.
 */
int append_image_data_csv(char* filename, char* image_filename, std::vector<double>& image_data, int reset_file)
{
    char buffer[256];
    char mode[8];
    FILE* fp;

    strcpy_s(mode, "a");

    if (reset_file)
    {
        strcpy_s(mode, "w");
    }

    fp = fopen(filename, mode);
    if (!fp)
    {
        printf("Unable to open output file %s\n", filename);
        exit(-1);
    }

    // write the filename and the feature vector to the CSV file
    strcpy_s(buffer, image_filename);
    std::fwrite(buffer, sizeof(char), strlen(buffer), fp);
    for (int i = 0; i < image_data.size(); i++)
    {
        char tmp[256];
        sprintf_s(tmp, ",%.12f", image_data[i]);
        std::fwrite(tmp, sizeof(char), strlen(tmp), fp);
    }

    std::fwrite("\n", sizeof(char), 1, fp); // EOL

    fclose(fp);

    return (0);
}


int main() {
    // Set up video capture object, e.g.:
    VideoCapture cap(0); // Capture video from the default camera

    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    vector<Mat> templates;


    while (true) {
        Mat frame;
        cap >> frame; // Read the current frame

        if (frame.empty()) {
            break;
        }

        // Preprocess the frame
        Mat processedFrame = preprocessFrame(frame);

        // Threshold the frame
        Mat thresholdedFrame = thresholdFrame(processedFrame);


        // Clean up the thresholded frame
        //cleanImage(thresholdedFrame);

        // Display the original frame and thresholded frame
        imshow("Original Frame", frame);

        cv::Mat dilation_img;
        cv::Mat d_cleanup_img;
        cv::Mat erosion_img;
        cv::Mat e_cleanup_img;
        char dilation[] = "dilation";
        char erosion[] = "erosion";
        // clean up images

        Mat distTransformed;
        distanceTransform(thresholdedFrame, distTransformed, DIST_L2, 3);



        // Threshold the distance transformed image to obtain a binary image
        Mat distTransformedBinary;
        threshold(distTransformed, distTransformedBinary, 0.5, 1, THRESH_BINARY_INV);

        // Convert the binary image to CV_8U type to apply morphological operations
        distTransformedBinary.convertTo(distTransformedBinary, CV_8U, 255);

        // Perform morphological operations: erode and dilate
        morph_cleanup(distTransformedBinary, d_cleanup_img, dilation);
        morph_cleanup(d_cleanup_img, e_cleanup_img, erosion);

        // Perform segmentation

        std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> segmented_result = Segmentation(e_cleanup_img);
        cv::Mat segmentedFrame = std::get<0>(segmented_result);
        cv::Mat labels = std::get<1>(segmented_result);
        cv::Mat stats = std::get<2>(segmented_result);
        cv::Mat centroids = std::get<3>(segmented_result);

        // Get the list of region IDs sorted by area
        std::vector<std::pair<int, int>> regions;
        for (int i = 1; i < centroids.rows; ++i) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            regions.push_back(std::make_pair(i, area));
        }
        std::sort(regions.begin(), regions.end(), [](std::pair<int, int> a, std::pair<int, int> b) {
            return a.second > b.second;
            });

        // Compute and display features for the largest N regions
        const int N = 3; // Limit recognition to the largest N regions
       // Inside the main loop
        std::vector<double> lastFeatureVector;
        std::vector<std::vector<double>> allFeatureVectors;
        for (int i = 0; i < std::min(N, (int)regions.size()); ++i) {
            int region_id = regions[i].first;
            int region_area = regions[i].second;
            if (region_area < 1000) {
                continue; // Ignore regions that are too small
            }
            std::vector<double> featureVector = computeFeatures(frame, labels, region_id);
            allFeatureVectors.push_back(featureVector);
            lastFeatureVector = featureVector;
        }
        cv::imshow("Frame with Features", frame);

        //Part 5: 
        char key = cv::waitKey(10);
        char img_folder[512] = "data/";
        // save the features of image to csv file
        if (key == 'n')
        {
            char* object_name = new char[256];
            // ask user to enter the name of the object
            std::cout << "Enter the category of the object: ";
            std::cin.getline(object_name, 256);
            std::cout << std::endl;
            // make the object name to the classifing name for later processing
            strcat_s(img_folder, object_name);
            strcat_s(img_folder, ".csv");

            // store data into csv file
            append_image_data_csv(img_folder, object_name, lastFeatureVector, 0);
            //calc_std();
        }

        // Mat featureFrame = featureComp(e_cleanup_img);


        //Part 3
        //imshow("Segmented Frame", segmentedFrame);
        if (waitKey(1) == 27) { // Wait for 1ms and check if the escape key is pressed
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}