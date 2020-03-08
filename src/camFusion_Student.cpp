
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left, bottom+20), cv::FONT_ITALIC, 0.5, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left, bottom+40), cv::FONT_ITALIC, 0.5, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
  	std::vector<cv::DMatch> kptMatches_ROI;
    //if the keypoint is within the bounding box, then insert it into ROI vector "kptMatches_ROI"
    for (auto it1=kptMatches.begin(); it1<kptMatches.end(); ++it1)
    {
      if (boundingBox.roi.contains(kptsCurr[it1->trainIdx].pt))
      {
        kptMatches_ROI.push_back(*it1);
      }
    }
  cout << "numer:" << kptMatches_ROI.size()<<endl;
  	//compute a robust mean of all the euclidean distances between keypoint matches 
  	double dis_sum = 0;
  	for (auto it2=kptMatches_ROI.begin(); it2<kptMatches_ROI.end(); ++it2)
    {
      //double distance = cv::norm(kptsCurr[it2->trainIdx].pt - kptsPrev[it2->queryIdx].pt);
      //dis_sum = dis_sum + distance;
      dis_sum = dis_sum + it2->distance;
    }
    double dis_mean = dis_sum / kptMatches_ROI.size();    //calculate the mean distance of all of the matches.
  	
  
  	//then remove those that are too far away from the mean.
  	for (auto it3=kptMatches_ROI.begin(); it3<kptMatches_ROI.end(); ++it3)
    {
      //double distance = cv::norm(kptsCurr[it3->trainIdx].pt - kptsPrev[it3->queryIdx].pt);
      if ((it3->distance > dis_mean*0.8) && (it3->distance < dis_mean*1.2))
      {
        boundingBox.kptMatches.push_back(*it3);
      }
    }
  cout << "numer:" << boundingBox.kptMatches.size()<<endl;

}
https://github.com/xiaohuieddie/SFND-3D-Object-Tracking

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
         
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

  	std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 1/frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
      	if (it->y > (-laneWidth/2) && it->y < (laneWidth/2)){
          minXPrev = minXPrev > it->x ? it->x : minXPrev;
        }
      	else{
          continue;
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
      if (it->y > (-laneWidth/2) && it->y < (laneWidth/2)){
          minXCurr = minXCurr > it->x ? it->x : minXCurr;
        }
      	else{
          continue;
        }  
      
    }

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
  	std::multimap<int, int> Keypoint_BoxID;
    int maxPrevBoxID = 0;
  
  	//// Search for the box ID for each matched keypoint in both previous and current frame
    for (auto it1=matches.begin(); it1<matches.end(); ++it1)
    {
      //Locate the keypoints for each pair of match
      cv::KeyPoint Prev_Keypoint = prevFrame.keypoints[it1->queryIdx];
      cv::KeyPoint Curr_Keypoint = currFrame.keypoints[it1->trainIdx];
      
      //Initianize the BoxID 
      int Prev_BoxID;
      int Curr_BoxID;   
      bool Prev_Find = false;
      bool Curr_Find = false;
      
      //Find the BoxID of the keypoint from previous frame
      for (auto it2=prevFrame.boundingBoxes.begin(); it2<prevFrame.boundingBoxes.end(); ++it2)
      {
      	if (it2->roi.contains(Prev_Keypoint.pt))
        {
          Prev_BoxID = it2->boxID;
          Prev_Find = true;
          break;
        }
      }
            
      //Find the BoxID of the keypoint from current frame
      for (auto it3=currFrame.boundingBoxes.begin(); it3<currFrame.boundingBoxes.end(); ++it3)
      {
      	if (it3->roi.contains(Curr_Keypoint.pt))
        {
          Curr_BoxID = it3->boxID;
          Curr_Find = true;
          break;
        }
      }
      
      if (Prev_Find && Curr_Find)
      {
        maxPrevBoxID = std::max(maxPrevBoxID, Prev_BoxID);      
      	Keypoint_BoxID.insert({Curr_BoxID, Prev_BoxID});
      }
      else{
        continue;
      }
    }
       
  	cout << "Done 1"<<endl;
    //// Loop through each boxID in the current frame, and get the mode (most frequent value) of associated boxID for the previous frame
    for (auto itr = currFrame.boundingBoxes.begin(); itr < currFrame.boundingBoxes.end(); ++itr)
    {
      int curr_ID = itr->boxID;
      std::multimap<int,int>::iterator itr1;   //create a multimap iterator
      std::vector<int> count (maxPrevBoxID+1, 0);    //create a vertor will all of the elements being 0. The length of this vector is the max ID number of previous bounding boxes.
      
      // Accumulator loop
      for (itr1=Keypoint_BoxID.equal_range(curr_ID).first; itr1 != Keypoint_BoxID.equal_range(curr_ID).second; ++itr1)
      {
        count[(*itr1).second] += 1;
      }
      
      // Get the index of the maximum count (the mode) of the previous frame's boxID
      int modeIndex = std::distance(count.begin(), std::max_element(count.begin(), count.end()));
     
      bbBestMatches.insert({modeIndex, curr_ID});   
      cout << "Current Box ID Done " << curr_ID << "; Prev Box ID Done " << modeIndex << endl;
    }
            
}
