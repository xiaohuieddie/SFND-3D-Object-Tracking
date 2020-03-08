
#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <opencv2/core.hpp>
#include "dataStructures.h"

#include <fstream>
#include <iomanip>
#include <cmath>
#include <limits>


void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT);
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches);
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame);

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait=true);

void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg=nullptr);
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC);
                     
void DataLog(std::vector<std::string> detectorType_list, std::vector<std::string> descriptorType_list, int imgStartIndex, int imgEndIndex, int imgStepWidth, bool data_save);     

double Detector(std::string detectorType, std::vector<cv::KeyPoint> &keypoints, cv::Mat &imgGray, bool bVis);

std::vector<double> Test(int imgIndex, std::vector<DataFrame> &dataBuffer, std::string detectorType, std::string descriptorType);
#endif /* camFusion_hpp */
