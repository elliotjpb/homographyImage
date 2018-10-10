#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv::xfeatures2d;
using namespace std;
using namespace cv;
//To get homography from images passed in. Matching points in the images.

Mat Homography(Mat image1, Mat image2){
    Mat I_1 = image1;
    Mat I_2 = image2;
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

    	// Step 1: Detect the keypoints:
    	std::vector<KeyPoint> keypoints_1, keypoints_2;
    	f2d->detect( I_1, keypoints_1 );
    	f2d->detect( I_2, keypoints_2 );
    	// Step 2: Calculate descriptors (feature vectors)
    	Mat descriptors_1, descriptors_2;
    	f2d->compute( I_1, keypoints_1, descriptors_1 );
    	f2d->compute( I_2, keypoints_2, descriptors_2 );
    	// Step 3: Matching descriptor vectors using BFMatcher :
    	BFMatcher matcher;
    	std::vector< DMatch > matches;
    	matcher.match( descriptors_1, descriptors_2, matches );
    	// Keep best matches only to have a nice drawing.
    	// We sort distance between descriptor matches
    	Mat index;
    	int nbMatch = int(matches.size());
    	Mat tab(nbMatch, 1, CV_32F);
    	for (int i = 0; i < nbMatch; i++)
    		tab.at<float>(i, 0) = matches[i].distance;
    	sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
    	vector<DMatch> bestMatches;

    	for (int i = 0; i < 200; i++)
    		bestMatches.push_back(matches[index.at < int > (i, 0)]);
    	// 1st image is the destination image and the 2nd image is the src image
    	std::vector<Point2f> dst_pts;                   //1st
    	std::vector<Point2f> source_pts;                //2nd

    	for (vector<DMatch>::iterator it = bestMatches.begin(); it != bestMatches.end(); ++it) {
    		//cout << it->queryIdx << "\t" <<  it->trainIdx << "\t"  <<  it->distance << "\n";
    		//-- Get the keypoints from the good matches
    		dst_pts.push_back( keypoints_1[ it->queryIdx ].pt );
    		source_pts.push_back( keypoints_2[ it->trainIdx ].pt );
    	}

    	Mat H = findHomography( source_pts, dst_pts, CV_RANSAC );
    	//cout << H_12 << endl;
      return H;
}

void readme(){
    std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl;
}

/** @function main */
int main(int argc, char** argv){

  if (argc != 5){
      readme(); return -1;
  }

  Mat im_1 = imread(argv[1], IMREAD_COLOR);
  Mat im_2 = imread(argv[2], IMREAD_COLOR);
  Mat im_3 = imread(argv[3], IMREAD_COLOR);
  Mat im_4 = imread(argv[4], IMREAD_COLOR);

  if (!im_1.data || !im_2.data || !im_3.data || !im_4.data){
      std::cout << " --(!) Error reading images " << std::endl; return -1;
  }

   //variable to store homography
   Mat H_12, H_23, H_34;
   //Finding homography of all images
   H_12 = Homography(im_1,im_2);
   H_23 = Homography(im_2,im_3);
   H_34 = Homography(im_3,im_4);

   std::cout << H_34 << '\n';

      Mat warpImage2;
      Mat warpImage3;
      Mat warpImage4;
      //warpPerspective(cap2frame, warpImage2, homography, Size(cap1frame.cols*2, cap1frame.rows*2), INTER_CUBIC);
      //warping the second video cap2frame so it matches with the first one.
      //size is defined as the final video size
      warpPerspective(im_2, warpImage2, H_12, Size(im_1.cols*2, im_1.rows*2), INTER_CUBIC);
      warpPerspective(im_3, warpImage3, H_23, Size(im_1.cols*2, im_1.rows*2), INTER_CUBIC);
      warpPerspective(im_4, warpImage4, H_34, Size(im_1.cols*2, im_1.rows*2), INTER_CUBIC);
      //std::cout << "cols " << (cap1frame.cols*2) << "rows " << (cap1frame.rows*2) << '\n';
      //final is the final canvas where both videos will be warped onto.
      Mat final(Size(im_1.cols*5 + im_1.cols, im_1.rows*5),CV_8UC3);
      //Mat final(Size(cap1frame.cols*2 + cap1frame.cols, cap1frame.rows*2),CV_8UC3);
      //Using roi getting the relivent areas of each video.
      Mat roi1(final, Rect(0, 0, im_1.cols, im_1.rows));
      Mat roi2(final, Rect(0, 0, warpImage2.cols, warpImage2.rows));
      Mat roi3(final, Rect(0, 0, warpImage3.cols, warpImage3.rows));
      Mat roi4(final, Rect(0, 0, warpImage4.cols, warpImage4.rows));
      //warping images on to the canvases which are linked with the final canvas.
      warpImage4.copyTo(roi4);
      warpImage3.copyTo(roi3);
      warpImage2.copyTo(roi2);
      im_1.copyTo(roi1);

      //int rows = final.rows;
      //int cols = final.cols;
      //std::cout << "rows " << rows << "cols " << cols << '\n';
      imshow ("Result", final);
      //imshow ("result", im_4);

    //if(waitKey(30) >= 0) break;
    //}
    waitKey(0);
    return 0;
}
