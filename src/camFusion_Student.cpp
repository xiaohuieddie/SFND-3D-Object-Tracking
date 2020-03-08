
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"

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

  	//compute a robust mean of all the euclidean distances between keypoint matches 
  	double dis_sum = 0;
  	for (auto it2=kptMatches_ROI.begin(); it2<kptMatches_ROI.end(); ++it2)
    {
      //double distance = cv::norm(kptsCurr[it2->trainIdx].pt - kptsPrev[it2->queryIdx].pt);
      //dis_sum = dis_sum + distance;
      dis_sum += it2->distance;
    }
    double dis_mean = dis_sum / kptMatches_ROI.size();    //calculate the mean distance of all of the matches.
  	 
  	//then remove those that are too far away from the mean.
  	for (auto it3=kptMatches_ROI.begin(); it3<kptMatches_ROI.end(); ++it3)
    {
      //double distance = cv::norm(kptsCurr[it3->trainIdx].pt - kptsPrev[it3->queryIdx].pt);
      if (it3->distance < dis_mean*1.1)
      {
        boundingBox.kptMatches.push_back(*it3);
      }
     }
}

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
    }           
}

//Log data of number of keypoints with regard to respective detector  
void DataLog(vector<string> detectorType_list, vector<string> descriptorType_list, int imgStartIndex, int imgEndIndex, int imgStepWidth, bool data_save)
{
  if (data_save){
    
    // Create the table format of the output "Data_Log.csv"
  	ofstream Camera_TTC, Lidar_TTC;
  	Camera_TTC.open("../Result/Camera_TTC.csv");
    Lidar_TTC.open("../Result/Lidar_TTC.csv");
    
    Camera_TTC << "Detector Type" << "," << "Descroptor Type" << ",";
    Lidar_TTC << "Detector Type" << "," << "Descroptor Type" << ",";
    
    for (int i=imgStartIndex+1; i<= imgEndIndex - imgStartIndex; i++)
    {
    	Camera_TTC << "Image" << i-1 << "-" << i << ","; 
        Lidar_TTC << "Image" << i-1 << "-" << i << ","; 
    }
    Camera_TTC << endl;
    Lidar_TTC << endl;
    
    // Loop each detector type
    for (auto it=detectorType_list.begin(); it<detectorType_list.end(); ++it)
    {
      // Loop each descriptor type 
      for (auto i=descriptorType_list.begin(); i<descriptorType_list.end(); ++i)
      {
		vector<DataFrame> dataBuffer;
        // AKAZE detector works only with AKAZE descriptor
        if ((i == descriptorType_list.begin()+4) && (it != detectorType_list.begin()+5)){
          cout << *it << " detector doesn't work with AKAZE descriptor" << endl;
          continue;
        }
        // SIFT detector and ORB descriptor do not work together
        else if((i == descriptorType_list.begin()+2) && (it == detectorType_list.begin()+6)){
          cout << *it << " detector doesn't work with " << *i << "descriptor" << endl;
          continue;
        }
        else{
          Camera_TTC << *it << "," << *i << "," ;
          Lidar_TTC  << *it << "," << *i << "," ;
          // Loop each image with regard to specific combination of detector and descriptor type
          for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
          {
            //// LOAD IMAGE INTO BUFFER 
			cout << "Working on Image #" << imgIndex << endl;
            vector<double> Tcc = Test(imgIndex, dataBuffer, *it, *i);
            if (dataBuffer.size() >1){
              if (Tcc[0]==0){
                Lidar_TTC << "N.A" <<',';
              }
              else{
              	Lidar_TTC << Tcc[0]<<',';
              }
              
              if (Tcc[1]==0){
                Camera_TTC << "N.A" <<',';
              }
              else{
              	Camera_TTC << Tcc[1]<<',';
              }
              
            }
            /*
            // assemble filenames for current index
            ostringstream imgNumber;
            imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
            string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

            // load image from file 
            cv::Mat img = cv::imread(imgFullFilename);

            // push image into data frame buffer
            DataFrame frame;
            frame.cameraImg = img;
            dataBuffer.push_back(frame);  
            cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

            // DETECT & CLASSIFY OBJECTS 
            float confThreshold = 0.2;
            float nmsThreshold = 0.4;        
            detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                          yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);
            cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;

            //// CROP LIDAR POINTS 
            // load 3D Lidar points from file
            string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
            std::vector<LidarPoint> lidarPoints;
            loadLidarFromFile(lidarPoints, lidarFullFilename);

            // remove Lidar points based on distance properties
            float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
            cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
            (dataBuffer.end() - 1)->lidarPoints = lidarPoints;
            cout << "#3 : CROP LIDAR POINTS done" << endl;

            //// CLUSTER LIDAR POINT CLOUD 
            // associate Lidar points with camera-based ROI
            float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
            clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);
            cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;

            //// DETECT IMAGE KEYPOINTS 
            // convert current image to grayscale
            cv::Mat imgGray;
            cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

            // extract 2D keypoints from current image
            vector<cv::KeyPoint> keypoints; // create empty feature list for current image        
            double t1 = Detector(*it, keypoints, imgGray, bVis);

            // push keypoints and descriptor for current frame to end of data buffer
            (dataBuffer.end() - 1)->keypoints = keypoints;
            cout<< "Number of keypoints identified for the preceding vehicle:" << keypoints.size() << endl;
            cout << "#5 : DETECT KEYPOINTS done" << endl;

            //// EXTRACT KEYPOINT DESCRIPTORS 
            cv::Mat descriptors;            
            double t2 = descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, *i);
            // push descriptors for current frame to end of data buffer
            (dataBuffer.end() - 1)->descriptors = descriptors;
            cout << "#6 : EXTRACT DESCRIPTORS done" << endl;

            if (dataBuffer.size() > 1) // wait until at least two images have been processed
            {
              //// MATCH KEYPOINT DESCRIPTORS 
              // Match descriptor
              string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
              string descriptorType_i = "DES_HOG"; // DES_BINARY, DES_HOG
              string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
              vector<cv::DMatch> matches;
              matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                               (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                               matches, descriptorType_i, matcherType, selectorType);

              // store matches in current data frame
              (dataBuffer.end() - 1)->kptMatches = matches;
              cout<< "Number of matched keypoints identified for the preceding vehicle:" << matches.size() << endl;
              cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;         
              //// TRACK 3D OBJECT BOUNDING BOXES 

              //// match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
              map<int, int> bbBestMatches;
              // associate bounding boxes between current and previous frame using keypoint matches
              matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); 
              // store matches in current data frame
              (dataBuffer.end()-1)->bbMatches = bbBestMatches;
              cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;

              //// COMPUTE TTC ON OBJECT IN FRONT 
              // loop over all BB match pairs
              for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
              {
                // find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB;
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                {
                  if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                  {
                    currBB = &(*it2);
                  }
                }

                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                {
                  if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                  {
                    prevBB = &(*it2);
                  }
                }

                // compute TTC for current match
                if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                {
                  //// compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                  double ttcLidar; 
                  computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                  cout << "Estimated time of Collision by Liar:"<< ttcLidar << endl;
                  Lidar_TTC << ttcLidar << ',';

                  //// assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                  //// compute time-to-collision based on camera (implement -> computeTTCCamera)
                  double ttcCamera;
                  clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);                    
                  computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
                  cout << "Estimated time of Collision by Camera:"<< ttcCamera << endl; 
                  Camera_TTC << ttcCamera << ',';

                } // eof TTC computation

              } // eof loop over all BB matches            

            }*/
		    cout << endl;
          } // eof loop over all images
          Camera_TTC << endl;
          Lidar_TTC << endl;
          cout << endl; 
        }
      } // end of descriptor type loop
    } // end of detector type loop
  }
}
 
double Detector(std::string detectorType, std::vector<cv::KeyPoint> &keypoints, cv::Mat &imgGray, bool bVis){
    //// -> SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    double t;
  	if (detectorType.compare("SHITOMASI") == 0)
    {
      t = detKeypointsShiTomasi(keypoints, imgGray, bVis);
    }
    else if (detectorType.compare("HARRIS") == 0)
    {
      t = detKeypointsHarris(keypoints, imgGray, bVis);
    }
    else
    {
      t = detKeypointsModern(keypoints, imgGray, detectorType, bVis);
    }
  
  	return t;
}

std::vector<double> Test(int imgIndex, vector<DataFrame> &dataBuffer, string detectorType, string descriptorType){
  	vector<double> TTC;
  // data location
    string dataPath = "../";

  	// detector and descriptor type
  	string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
  	string descriptorType_i = "DES_HOG"; // DES_BINARY, DES_HOG
  	string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN
  	
    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    //vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results
 /* LOAD IMAGE INTO BUFFER */

    // assemble filenames for current index
    ostringstream imgNumber;
    imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
    string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

    // load image from file 
    cv::Mat img = cv::imread(imgFullFilename);

    // push image into data frame buffer
    DataFrame frame;
    frame.cameraImg = img;
    dataBuffer.push_back(frame);

    cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


    /* DETECT & CLASSIFY OBJECTS */

    float confThreshold = 0.2;
    float nmsThreshold = 0.4;        
    detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                  yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

    cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


    /* CROP LIDAR POINTS */

    // load 3D Lidar points from file
    string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
    std::vector<LidarPoint> lidarPoints;
    loadLidarFromFile(lidarPoints, lidarFullFilename);

    // remove Lidar points based on distance properties
    float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
    cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);   
    (dataBuffer.end() - 1)->lidarPoints = lidarPoints;
    cout << "#3 : CROP LIDAR POINTS done" << endl;

    /* CLUSTER LIDAR POINT CLOUD */
    // associate Lidar points with camera-based ROI
    float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
    clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

    // Visualize 3D objects
    bVis = false;
    if(bVis)
    {
      show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(10.0, 25.0), cv::Size(700, 600), true);
    }
    bVis = false;

    cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
    cout << "number of bounding box: " << (dataBuffer.end()-1)->boundingBoxes.size() << endl;

    // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
    //continue; // skips directly to the next image without processing what comes beneath

    /* DETECT IMAGE KEYPOINTS */

    // convert current image to grayscale
    cv::Mat imgGray;
    cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

    // extract 2D keypoints from current image
    vector<cv::KeyPoint> keypoints; // create empty feature list for current image        
    double t1 = Detector(detectorType, keypoints, imgGray, bVis);      	

    // optional : limit number of keypoints (helpful for debugging and learning)
    bool bLimitKpts = false;
    if (bLimitKpts)
    {
      int maxKeypoints = 50;

      if (detectorType.compare("SHITOMASI") == 0)
      { // there is no response info, so keep the first 50 as they are sorted in descending quality order
        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
      }
      cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
      cout << " NOTE: Keypoints have been limited!" << endl;
    }

    // push keypoints and descriptor for current frame to end of data buffer
    (dataBuffer.end() - 1)->keypoints = keypoints;
    cout<< "Number of keypoints identified for the preceding vehicle:" << keypoints.size() << endl;
    cout << "#5 : DETECT KEYPOINTS done" << endl;


    /* EXTRACT KEYPOINT DESCRIPTORS */
    cv::Mat descriptors;
    double t2 = descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

    // push descriptors for current frame to end of data buffer
    (dataBuffer.end() - 1)->descriptors = descriptors;
    cout << "#6 : EXTRACT DESCRIPTORS done" << endl;

    if (dataBuffer.size() > 1) // wait until at least two images have been processed
    {
      /* MATCH KEYPOINT DESCRIPTORS */
      vector<cv::DMatch> matches;
      matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                       (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                       matches, descriptorType_i, matcherType, selectorType);

      // store matches in current data frame
      (dataBuffer.end() - 1)->kptMatches = matches;
      cout<< "Number of matched keypoints identified for the preceding vehicle:" << matches.size() << endl;
      cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;


      /* TRACK 3D OBJECT BOUNDING BOXES */

      //// STUDENT ASSIGNMENT
      //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
      map<int, int> bbBestMatches;
      matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
      //// EOF STUDENT ASSIGNMENT
      for (auto itr = bbBestMatches.begin(); itr != bbBestMatches.end(); ++itr) { 
        cout << itr->first << '\t' << itr->second << '\n'; 
      }
      // store matches in current data frame
      (dataBuffer.end()-1)->bbMatches = bbBestMatches;

      cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;
      //continue;

      /* COMPUTE TTC ON OBJECT IN FRONT */

      // loop over all BB match pairs
      for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
      {
        // find bounding boxes associates with current match
        BoundingBox *prevBB, *currBB;
        for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
        {
          if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
          {
            currBB = &(*it2);
          }
        }

        for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
        {
          if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
          {
            prevBB = &(*it2);
          }
        }

        // compute TTC for current match
        if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
        {
          //// STUDENT ASSIGNMENT
          //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
          
          double ttcLidar; 
          computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
          cout << "Estimated time of Collision by Liar:"<< ttcLidar << endl;
          TTC.push_back(ttcLidar);
          //// EOF STUDENT ASSIGNMENT

          //// STUDENT ASSIGNMENT
          //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
          //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
          double ttcCamera;
          clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);                    
          computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);

          cout << "Estimated time of Collision by Camera:"<< ttcCamera << endl;
          TTC.push_back(ttcCamera);
          
          
          //// EOF STUDENT ASSIGNMENT


        } // eof TTC computation
        
        
      } // eof loop over all BB matches            

    } 
  if (TTC.size()==0){
   	TTC.push_back(0); 
    TTC.push_back(0); 
  }
  return TTC;
}