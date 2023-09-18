#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;


//Part 1:
void extract_features_baseline(const Mat& img, vector<float>& features) {
    Mat_<Vec3b> imgVec = img;
    features.clear(); 
    int r0 = img.rows / 2 - 5;
    int c0 = img.cols / 2 - 5;
    int d, c, r;
    for (d = 0; d < img.channels(); d++) {
        for (r = 0; r < 9; r++) {
            for (c = 0; c < 9; c++) {
                features.push_back((float)imgVec(r + r0, c + c0)[d]);
            }
        }
    }
}

float distance_baseline(const vector<float>& a, const vector<float>& b) {
    if (a.size() != b.size()) {
        return 0;
    }
    float distance = 0.0;
    for (int i = 0; i < a.size(); i++) {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return distance;
}

//Part 2:
int extract_features_histogram(cv::Mat& image, std::vector<float>& feature_vector, bool center) {
    int num_bins = 8 * 8 * 8;
    feature_vector = std::vector<float>(num_bins, 0.0f);
    int r = center ? image.rows / 3 : 0;
    int c = image.cols / 3;
    float inv_denom = 1.0f / ((image.rows - 2 * r) * (image.cols - 2 * c));

    for (int i = 0 + r; i < image.rows - r; i++) {
        cv::Vec3b* pixel_ptr = image.ptr<cv::Vec3b>(i);
        for (int j = 0 + c; j < image.cols - c; j++) {
            int bin0 = pixel_ptr[j][0] / 32;
            int bin1 = pixel_ptr[j][1] / 32;
            int bin2 = pixel_ptr[j][2] / 32;
            int bin_idx = bin0 * 8 * 8 + bin1 * 8 + bin2;
            feature_vector[bin_idx] += 1;
        }
    }

    for (int i = 0; i < feature_vector.size(); i++) {
        feature_vector[i] *= inv_denom;
    }
    return 0;
}

float distance_histogram(const vector<float>& hist_a, const vector<float>& hist_b) {
    float intersection_sum = 0;
    int hist_size = hist_a.size();
    //cout << hist_size;

    for (int i = 0; i < hist_size; i++) {
        intersection_sum += min(hist_a[i], hist_b[i]);
    }

    return 1.0f - intersection_sum;
}

//Part 3: Multi histograms
int extract_features_multihistogram(Mat& src, vector<float>& feature) {
    vector<float> wholeFeature;
    vector<float> centerFeature;

    extract_features_histogram(src, wholeFeature,false);
    extract_features_histogram(src, centerFeature, true);

    // initialize feature vector
    feature = {};
    // combine two feature vectors
    feature.insert(feature.end(), wholeFeature.begin(), wholeFeature.end());
    feature.insert(feature.end(), centerFeature.begin(), centerFeature.end());
    
    return 0;
}


float distance_multihistogram(const vector<float>& target, const vector<float>& candidate) {
    float wholeDis = 0;  // the distance between whole-image histograms
    float centerDis = 0; // the distance between center-block histograms
    float weight = 0.4;  // the weight of wholeDis, between 0 and 1

    int size = candidate.size() / 2;
    for (int i = 0; i < size; i++) {
        wholeDis += abs(target[i] - candidate[i]);
    }

    for (int i = size; i < candidate.size(); i++) {
        centerDis += abs(target[i] - candidate[i]);
    }

    // Get the biased distance
    float distance = (weight * wholeDis + (1 - weight) * centerDis) / candidate.size();

    return distance;
}

//Part 4: Texture Histograms 
void extract_features_texture_histogram(const Mat& image, vector<float>& feature_vector, int num_bins = 64) {
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    Mat grad_x, grad_y;
    Sobel(gray_image, grad_x, CV_32F, 1, 0);
    Sobel(gray_image, grad_y, CV_32F, 0, 1);

    Mat grad_mag;
    magnitude(grad_x, grad_y, grad_mag);

    feature_vector = vector<float>(num_bins, 0.0f);
    float range_max = 256.0f;
    float bin_width = range_max / num_bins;

    for (int i = 0; i < grad_mag.rows; i++) {
        for (int j = 0; j < grad_mag.cols; j++) {
            int bin_idx = static_cast<int>(grad_mag.at<float>(i, j) / bin_width);
            if (bin_idx >= num_bins) {
                bin_idx = num_bins - 1;
            }
            feature_vector[bin_idx] += 1;
        }
    }

    float inv_denom = 1.0f / (grad_mag.rows * grad_mag.cols);
    for (int i = 0; i < feature_vector.size(); i++) {
        feature_vector[i] *= inv_denom;
    }
}

int extract_features_combined_histogram(Mat& src, vector<float>& feature) {

    vector<float> colorWholeFeature;
    vector<float> colorCenterFeature;
    vector<float> textureFeature;

    extract_features_histogram(src, colorWholeFeature, false);
    extract_features_histogram(src, colorCenterFeature, true);
    extract_features_texture_histogram(src, textureFeature);

    feature = {};
    feature.insert(feature.end(), colorWholeFeature.begin(), colorWholeFeature.end());
    feature.insert(feature.end(), colorCenterFeature.begin(), colorCenterFeature.end());
    feature.insert(feature.end(), textureFeature.begin(), textureFeature.end());

    return 0;
}

float distance_texture_histogram(const vector<float>& hist_a, const vector<float>& hist_b) {
    float intersection_sum = 0;
    int hist_size = hist_a.size();
    
    for (int i = 0; i < hist_size; i++) {
        
        intersection_sum += min(hist_a[i], hist_b[i]);
    }
    return 1.0f - intersection_sum;
}


//Part 5: Custom CBIR (Combining extract_features_histogram, Shape Feature: Hu Moments histogram and Texture Feature: Local Binary Patterns (LBP) histogram):
void extract_hu_moments(const cv::Mat& image, std::vector<double>& hu_moments) {
    cv::Mat gray_image, binary_image;

    // Convert the image to grayscale
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    // Threshold the image to create a binary image
    cv::threshold(gray_image, binary_image, 128, 255, cv::THRESH_BINARY);

    // Compute moments
    cv::Moments moments = cv::moments(binary_image, true);

    // Compute Hu moments
    cv::HuMoments(moments, hu_moments);
}

void compute_lbp(const cv::Mat& image, cv::Mat& lbp) {
    lbp = cv::Mat::zeros(image.rows - 2, image.cols - 2, CV_8UC1);

    for (int i = 1; i < image.rows - 1; i++) {
        for (int j = 1; j < image.cols - 1; j++) {
            uchar center = image.at<uchar>(i, j);
            uchar code = 0;
            code |= (image.at<uchar>(i - 1, j - 1) >= center) << 7;
            code |= (image.at<uchar>(i - 1, j) >= center) << 6;
            code |= (image.at<uchar>(i - 1, j + 1) >= center) << 5;
            code |= (image.at<uchar>(i, j + 1) >= center) << 4;
            code |= (image.at<uchar>(i + 1, j + 1) >= center) << 3;
            code |= (image.at<uchar>(i + 1, j) >= center) << 2;
            code |= (image.at<uchar>(i + 1, j - 1) >= center) << 1;
            code |= (image.at<uchar>(i, j - 1) >= center) << 0;

            lbp.at<uchar>(i - 1, j - 1) = code;
        }
    }
}

void extract_lbp_histogram(const cv::Mat& image, std::vector<float>& lbp_histogram, int num_bins = 256) {
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    cv::Mat lbp_image;
    compute_lbp(gray_image, lbp_image);

    int histSize[] = { num_bins };
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    cv::Mat hist;

    cv::calcHist(&lbp_image, 1, 0, cv::Mat(), hist, 1, histSize, histRange);
    hist.convertTo(hist, CV_32F);
    hist = hist / (lbp_image.rows * lbp_image.cols);

    lbp_histogram = std::vector<float>(hist.begin<float>(), hist.end<float>());
}

void extract_CBIR_features(cv::Mat& image, std::vector<float>& feature_vector) {
    std::vector<float> color_histogram;
    std::vector<double> hu_moments;
    std::vector<float> lbp_histogram;

    // Extract features
    extract_features_histogram(image, color_histogram, false);
    extract_hu_moments(image, hu_moments);
    extract_lbp_histogram(image, lbp_histogram);

    // Normalize each feature component
    cv::normalize(color_histogram, color_histogram, 1, 0, cv::NORM_L1);
    cv::normalize(hu_moments, hu_moments, 1, 0, cv::NORM_L1);
    cv::normalize(lbp_histogram, lbp_histogram, 1, 0, cv::NORM_L1);

    // Combine the features into one feature vector
    feature_vector.clear();
    feature_vector.insert(feature_vector.end(), color_histogram.begin(), color_histogram.end());
    feature_vector.insert(feature_vector.end(), hu_moments.begin(), hu_moments.end());
    feature_vector.insert(feature_vector.end(), lbp_histogram.begin(), lbp_histogram.end());
}

float distance_combined_histogram(const std::vector<float>& color_hist_a, const std::vector<float>& color_hist_b, float color_weight,
    const std::vector<double>& hu_moments_a, const std::vector<double>& hu_moments_b, float hu_weight,
    const std::vector<float>& lbp_hist_a, const std::vector<float>& lbp_hist_b, float lbp_weight) {
    float intersection_sum_color = 0;
    float intersection_sum_hu = 0;
    float intersection_sum_lbp = 0;

    int color_hist_size = color_hist_a.size();
    int hu_hist_size = hu_moments_a.size();
    int lbp_hist_size = lbp_hist_a.size();

    for (int i = 0; i < color_hist_size; i++) {
        intersection_sum_color += min(color_hist_a[i], color_hist_b[i]);
    }

    for (int i = 0; i < hu_hist_size; i++) {
        intersection_sum_hu += min(hu_moments_a[i], hu_moments_b[i]);
    }

    for (int i = 0; i < lbp_hist_size; i++) {
        intersection_sum_lbp += min(lbp_hist_a[i], lbp_hist_b[i]);
    }

    float distance = (color_weight * (1.0f - intersection_sum_color)) +
        (hu_weight * (1.0f - intersection_sum_hu)) +
        (lbp_weight * (1.0f - intersection_sum_lbp));

    return distance;
}



int main(int argc, char* argv[]) {
    if (argc != 5) {
        cout << "Usage: " << argv[0] << " <target_image> <image_directory> <N_results> <matching_type>" << endl;
        return 1;
    }

    string target_image_path = argv[1];
    string image_directory = argv[2];
    int N_results = stoi(argv[3]);
    string matching_type = argv[4];

    Mat target_image = imread(target_image_path, IMREAD_COLOR);
    if (target_image.empty()) {
        cout << "Error: Could not load target image." << endl;
        return 1;
    }

    vector<float> target_features;

    if (matching_type == "baseline") {
        extract_features_baseline(target_image, target_features);
    }
    else if (matching_type == "histogram") {
        extract_features_histogram(target_image, target_features,true);
    }
    else if (matching_type == "multihistogram") {
        extract_features_multihistogram(target_image, target_features);
    }
    else if (matching_type == "combinedhistogram") {
        extract_features_combined_histogram(target_image, target_features);
    }
    else if (matching_type == "CBIR") {
        extract_CBIR_features(target_image, target_features);
    }
    else {
        cout << "Error: Invalid matching type." << endl;
        return 1;
    }

    using DistPair = pair<float, string>;
    auto comp = [](const DistPair& a, const DistPair& b) { return a.first < b.first; };
    priority_queue<DistPair, vector<DistPair>, decltype(comp)> distances(comp);

    DIR* dirp;
    struct dirent* dp;
    dirp = opendir(image_directory.c_str());

    if (dirp == NULL) {
        cout << "Error: Cannot open directory " << image_directory << endl;
        return 1;
    }



    while ((dp = readdir(dirp)) != NULL && dirp != NULL) {
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif")) {
            string img_path = image_directory + "/" + dp->d_name;

            if (img_path == target_image_path) {
                continue;
            }

            Mat img = imread(img_path, IMREAD_COLOR);
            if (img.empty()) {
                continue;
            }

            vector<float> candidate_features;
            if (matching_type == "baseline") {
                extract_features_baseline(img, candidate_features);
            }
            else if (matching_type == "histogram") {
                extract_features_histogram(img, candidate_features,false);
            }
            else if (matching_type == "multihistogram") {
                extract_features_multihistogram(img, candidate_features);
            }
            else if (matching_type == "combinedhistogram") {
                extract_features_combined_histogram(img, candidate_features);
            }
            else if (matching_type == "CBIR") {
                extract_CBIR_features(img, candidate_features);
            }
            
            float dist;
            if (matching_type == "baseline") {
                dist = distance_baseline(target_features, candidate_features);
            }
            else if (matching_type == "histogram") {
                dist = distance_histogram(target_features, candidate_features);
            }
            else if (matching_type == "multihistogram") {
                dist = distance_multihistogram(target_features, candidate_features);
            }

            else if (matching_type == "combinedhistogram") {
                int size = candidate_features.size();
                int colorWholeSize = size / 4;
                int colorCenterSize = size / 2;
                int textureSize = size / 4;
                //cout << size;
                float colorWholeDis = distance_histogram(
                    vector<float>(target_features.begin(), target_features.begin() + colorWholeSize),
                    vector<float>(candidate_features.begin(), candidate_features.begin() + colorWholeSize));

                float colorCenterDis = distance_histogram(
                    vector<float>(target_features.begin() + colorWholeSize, target_features.begin() + colorCenterSize),
                    vector<float>(candidate_features.begin() + colorWholeSize, candidate_features.begin() + colorCenterSize));

                float textureDis = distance_texture_histogram(
                    vector<float>(target_features.begin() + colorCenterSize, target_features.end()),
                    vector<float>(candidate_features.begin() + colorCenterSize, candidate_features.end()));

                dist = 0.4f * colorWholeDis + 0.4f * colorCenterDis + 0.2f * textureDis;
            }
            else if (matching_type == "CBIR") {
                float color_weight = 0.4f;
                float hu_weight = 0.3f;
                float lbp_weight = 0.3f;

                int color_hist_size = target_features.size() / 2;
                int hu_hist_size = 7;

                dist = distance_combined_histogram(
                    vector<float>(target_features.begin(), target_features.begin() + color_hist_size),
                    vector<float>(candidate_features.begin(), candidate_features.begin() + color_hist_size),
                    color_weight,
                    vector<double>(target_features.begin() + color_hist_size, target_features.begin() + color_hist_size + hu_hist_size),
                    vector<double>(candidate_features.begin() + color_hist_size, candidate_features.begin() + color_hist_size + hu_hist_size),
                    hu_weight,
                    vector<float>(target_features.begin() + color_hist_size + hu_hist_size, target_features.end()),
                    vector<float>(candidate_features.begin() + color_hist_size + hu_hist_size, candidate_features.end()),
                    lbp_weight);
            }
    
           
            if (distances.size() < N_results) {
                distances.push(make_pair(dist, img_path));
            }
            else {
                if (dist < distances.top().first) {
                    distances.pop();
                    distances.push(make_pair(dist, img_path));
                }
            }
        }
    }

    closedir(dirp);

    vector<DistPair> sorted_distances;
    while (!distances.empty()) {
        sorted_distances.push_back(distances.top());
        distances.pop();
    }

    reverse(sorted_distances.begin(), sorted_distances.end());

    cout << "Top " << N_results << " matches:" << endl;
    for (int i = 0; i < N_results && i < sorted_distances.size(); ++i) {
        cout << sorted_distances[i].second << " (distance: " << sorted_distances[i].first << ")" << endl;
    }

    return 0;
}

