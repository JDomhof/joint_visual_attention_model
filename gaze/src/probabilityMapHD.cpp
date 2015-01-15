 Skip to content
This repository

    Explore
    Gist
    Blog
    Help

    JDomhof JDomhof

    1
    0
    0

JDomhof/gaze

gaze/src/probabilityMapHD.cpp
JDomhof JDomhof 16 minutes ago
Create probabilityMapHD.cpp

1 contributor
661 lines (560 sloc) 24.53 kb
#include <Eigen/Dense>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>//Use cv_bridge to convert between ROS and OpenCV Image formats
#include <ros/ros.h>//Includes all the headers necessary to use the most common public pieces of the ROS system.
#include <sensor_msgs/image_encodings.h>//Include some useful constants for image encoding. Refer to: http://www.ros.org/doc/api/sensor_msgs/html/namespacesensor__msgs_1_1image__encodings.html for more info.
#include <opencv2/imgproc/imgproc.hpp>//Include headers for OpenCV Image processing
#include <opencv2/highgui/highgui.hpp>//Include headers for OpenCV GUI handling
#include "opencv2/core/core.hpp"
#include "opencv/cv.h"
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/core/eigen.hpp>
#include <image_transport/image_transport.h>//Use image_transport for publishing and subscribing to images in ROS
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <aruco/aruco.h>
#include <aruco/cvdrawingutils.h>
#include <numeric>
using namespace cv;
using namespace std;
using namespace aruco;
class CreateProbabilityMapGaze {
public:
ros::Publisher pub;
ros::Subscriber sub_image_kinect, sub_image_eyetracker, sub_gaze, sub_image_depth,sub_image_hd,sub_centroid;
ros::NodeHandle nh;
Point2f gaze_direction_eyetracker; //Coming from eyetracker, to be transformed to Kinect frame
Point2f gaze_direction_kinect;
Point2f maxLocationFiltering;
std::vector<Point2f> previous_gaze;
std::vector<Mat> maps;
std::vector<KeyPoint> keypoints_object, keypoints_scene;
int iWindowSize;
int iSizeGaussianBlurFilter;
Mat img_matches;
Mat imageKinect, imageEye,imageHD;
Mat imageProbabilityMap;
Mat depthMap;
geometry_msgs::PointStamped gaze_direction_out_msg;
cv_bridge::CvImagePtr cv_ptr_usb;
cv_bridge::CvImagePtr cv_ptr_kinect;
image_transport::Publisher pubimage;
bool b2Dviewer;
bool bUseHD;
int iMinimalMatches;
int iNumberOfMaps;
double dMeanDepthObjects;
cv::Mat M_extrinsic_HD_to_kinect, M_extrinsic_kinect_to_HD;
cv::Mat A_hdcamera , A_kinect;
CreateProbabilityMapGaze()
{
ros::NodeHandle private_nh("~");
private_nh.param("minimalMatches", iMinimalMatches, 20);
private_nh.param("useHD", bUseHD, true);
private_nh.param("SizeGaussianBlurFilter", iSizeGaussianBlurFilter, 101);
private_nh.param("TwoDviewer", b2Dviewer, false);
private_nh.param("NumberOfMaps", iNumberOfMaps, 1);
private_nh.param("WindowSize", iWindowSize, 150);
// Load stereo calibration data
{
std::string calib_data_location;
private_nh.getParam("stereoCalibrationData",calib_data_location);
cv::FileStorage fs(calib_data_location.c_str(),cv::FileStorage::READ);
if(!fs.isOpened()){ROS_ERROR("Camera Parameters file not found");};
fs["A_hdcamera"] >> A_hdcamera;
fs["A_kinect"] >> A_kinect;
fs["M_extrinsic_kinect_to_HD"] >> M_extrinsic_kinect_to_HD;
fs["M_extrinsic_HD_to_kinect"] >> M_extrinsic_HD_to_kinect;
ROS_INFO("Parameters loaded");
}
previous_gaze = std::vector<Point2f>(3, Point2f(0,0) );
maps = std::vector<Mat>(iNumberOfMaps, cv::Mat::zeros(imageKinect.size(), CV_8UC1) );
dMeanDepthObjects = 0;
pub = nh.advertise <geometry_msgs::PointStamped> ("/gaze_coordinates_target", 1);
image_transport::ImageTransport it(nh);
pubimage = it.advertise("/gaze/probabilitymap", 1);
sub_image_kinect = nh.subscribe("/camera/rgb/image_raw", 1, &CreateProbabilityMapGaze::callbackKinect, this);
sub_image_hd = nh.subscribe("/usb_cam/image_raw", 1, &CreateProbabilityMapGaze::callbackHD, this);
sub_image_eyetracker = nh.subscribe("/eyetracker/image", 1, &CreateProbabilityMapGaze::callbackEyetracker, this);
sub_gaze = nh.subscribe("/eyetracker/normalized_gaze_position", 1, &CreateProbabilityMapGaze::callbackGazeNormalized, this);
sub_image_depth = nh.subscribe("/segmenting/mask", 1, &CreateProbabilityMapGaze::callbackDepth, this);
sub_centroid = nh.subscribe("/segmenting/centroid_objects", 1, &CreateProbabilityMapGaze::callbackCentroid, this);
}
Point2f fromNormalizedToPixelCoordinates(geometry_msgs::PointStamped temp);
bool checkForMarkers();
void callbackHD(const sensor_msgs::ImageConstPtr& image1);
void callbackKinect(const sensor_msgs::ImageConstPtr& image1);
void callbackEyetracker(const sensor_msgs::ImageConstPtr& image2);
void callbackGazeNormalized(const geometry_msgs::PointStampedConstPtr& gaze);
void callbackGaze(const geometry_msgs::PointStampedConstPtr& gaze);
void callbackCentroid(const geometry_msgs::PointConstPtr& centroid);
void callbackDepth(const sensor_msgs::ImageConstPtr& image1);
void transformGaze();
std::vector< DMatch > keypointsKinect();
std::vector< DMatch > keypointsHD();
void backprojection();
Rect getROI(cv::Point Center,int iSizeWindow);
Point2f getCenter(cv::Point Center, int iSizeWindow);
};
void CreateProbabilityMapGaze::callbackHD(const sensor_msgs::ImageConstPtr& image1) {
cv_bridge::CvImagePtr cv_ptr_HD;
try
{
cv_ptr_HD = cv_bridge::toCvCopy(image1, sensor_msgs::image_encodings::BGR8);
}
catch (cv_bridge::Exception& e)
{
ROS_ERROR("Copy image failed", e.what());
return;
}
cv_ptr_HD->image.copyTo(imageHD);
}
void CreateProbabilityMapGaze::callbackKinect(const sensor_msgs::ImageConstPtr& image1) {
try
{
cv_ptr_kinect = cv_bridge::toCvCopy(image1, sensor_msgs::image_encodings::BGR8);
}
catch (cv_bridge::Exception& e)
{
ROS_ERROR("Copy image failed", e.what());
return;
}
cv_ptr_kinect->image.copyTo(imageKinect);
}
void CreateProbabilityMapGaze::callbackEyetracker(const sensor_msgs::ImageConstPtr& image2) {
try
{
cv_ptr_usb = cv_bridge::toCvCopy(image2, sensor_msgs::image_encodings::BGR8);
}
catch (cv_bridge::Exception& e)
{
ROS_ERROR("Copy image failed", e.what());
return;
}
cv_ptr_usb->image.copyTo(imageEye);
}
void CreateProbabilityMapGaze::callbackDepth(const sensor_msgs::ImageConstPtr& image1) {
cv_bridge::CvImagePtr cv_ptr;
try
{
cv_ptr = cv_bridge::toCvCopy(image1, sensor_msgs::image_encodings::BGR8);
}
catch (cv_bridge::Exception& e)
{
ROS_ERROR("Copy image failed", e.what());
return;
}
cv_ptr->image.copyTo(depthMap);
}
void CreateProbabilityMapGaze::callbackGazeNormalized(const geometry_msgs::PointStampedConstPtr& gaze) {
gaze_direction_eyetracker = fromNormalizedToPixelCoordinates(*gaze);
}
void CreateProbabilityMapGaze::callbackGaze(const geometry_msgs::PointStampedConstPtr& gaze) {
gaze_direction_eyetracker.x = (*gaze).point.x;
gaze_direction_eyetracker.y = (*gaze).point.y;
}
void CreateProbabilityMapGaze::callbackCentroid(const geometry_msgs::PointConstPtr& centroid) {
dMeanDepthObjects = (*centroid).z;
}
Point2f CreateProbabilityMapGaze::fromNormalizedToPixelCoordinates(geometry_msgs::PointStamped gaze) {
Point2f temp;
temp.x = (gaze).point.x * 1280;
temp.y = (1-(gaze).point.y) * 720;
return temp;
}
bool CreateProbabilityMapGaze::checkForMarkers() {
bool bMatchMarker = false;
MarkerDetector MDetector_kinect, MDetector_usb;
vector<Marker> Markers_kinect, Markers_usb;
MDetector_kinect.detect(imageKinect,Markers_kinect);
MDetector_usb.detect(imageEye,Markers_usb);
if (Markers_kinect.size() == 1 && Markers_usb.size()==1) {
if (Markers_kinect[0].id == Markers_usb[0].id) {
cout << "Match found with ID: " << Markers_kinect[0].id << endl;
bMatchMarker = true;
}
}
else {
cout << "No match found" << endl;
}
return bMatchMarker;
}
void CreateProbabilityMapGaze::transformGaze() {
if(!imageKinect.empty() && !imageEye.empty()) {
clock_t start_synchronizing = clock();
if(bUseHD && dMeanDepthObjects!=0) {
std::vector< DMatch > matches = keypointsHD();
if (matches.size() < iMinimalMatches) {
backprojection();
}
}
else {
std::vector< DMatch > matches = keypointsKinect();
if (matches.size() < iMinimalMatches) {
backprojection();
}
}
clock_t end_synchronizing = clock();
cout << "--> Synchronizing: " << (float)(end_synchronizing - start_synchronizing) / CLOCKS_PER_SEC << " seconds" << endl;
//Only when image exists
if (imageProbabilityMap.rows > 0 && imageProbabilityMap.cols > 0) {
if (b2Dviewer) {
circle(imageEye, gaze_direction_eyetracker, 10, Scalar(0,0,255), -1, 8 );
namedWindow("imageEye", CV_WINDOW_FREERATIO );
imshow("imageEye", imageEye);
resizeWindow("imageEye", 640, 480);
namedWindow("imageProbabilityMap", CV_WINDOW_FREERATIO );
imshow("imageProbabilityMap", imageProbabilityMap);
resizeWindow("imageProbabilityMap", 640, 480);
}
//Publish
sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", imageProbabilityMap).toImageMsg();
pubimage.publish(msg);
}
waitKey(3);
}
}
void CreateProbabilityMapGaze::backprojection() {
previous_gaze.erase(previous_gaze.begin());
previous_gaze.push_back(gaze_direction_eyetracker);
if (previous_gaze.at(0) != Point2f(0,0)) {
// downsample the image
Mat downsampled_half;
Mat downsampled;
//int sizeDownsampledTotal = 4;
//cv::pyrDown(imageEye, downsampled_half, cv::Size(imageEye.cols/2, imageEye.rows/2));
//cv::pyrDown(downsampled_half, downsampled, cv::Size(downsampled_half.cols/2, downsampled_half.rows/2));
int sizeDownsampledTotal = 2;
cv::pyrDown(imageEye, downsampled, cv::Size(imageEye.cols/2, imageEye.rows/2));
cv::Point2f zero(0.0f, 0.0f);
cv::Point2f sum = std::accumulate(previous_gaze.begin(), previous_gaze.end(), zero);
Point2f mean_point(sum.x / previous_gaze.size(), sum.y / previous_gaze.size());
Rect roi = getROI(Point(mean_point.x/sizeDownsampledTotal, mean_point.y/sizeDownsampledTotal), 50);
cout << "ROI: " << roi.x << ", " << roi.y << ", " << roi.width << ", " << roi.height << endl;
Mat maskGrabcut;
Mat bgModel,fgModel;
//In this case the foreground and backgroudn model are not used, possibly done by using gaze points.
grabCut( downsampled, maskGrabcut, roi, bgModel,fgModel, 1, cv::GC_INIT_WITH_RECT );
cv::compare(maskGrabcut,cv::GC_PR_FGD,maskGrabcut,cv::CMP_EQ);
cv::Mat foreground(downsampled.size(),CV_8UC3,cv::Scalar(255,255,255));
downsampled.copyTo(foreground,maskGrabcut); // bg pixels not copied
if (b2Dviewer) {
//Show foreground
//namedWindow("Foreground", CV_WINDOW_NORMAL);
//imshow("Foreground", foreground);
//resizeWindow("Foreground", 640, 480);
}
Mat backproj;
int iNumberOfChannels = 2;
if (iNumberOfChannels == 1) {
cout << "Hue and Saturation" << endl;
//---------------------Back Projection---------------------
Mat hsv_roi;
Mat hist;
cvtColor( foreground, hsv_roi, COLOR_BGR2HSV );
int histSize = MAX( 100, 2 );
float h_range[] = { 0, 179 };
const float* ranges[] = { h_range };
int channels[] = {0};
/// Get the Histogram and normalize it
calcHist( &hsv_roi, 1, channels, maskGrabcut, hist, 1, &histSize, ranges, true, false );
normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );
//
Mat hsv_target;
cvtColor( imageKinect, hsv_target, COLOR_BGR2HSV );
calcBackProject( &hsv_target, 1, channels, hist, backproj, ranges, 1, true );
//imshow("Back Projection", backproj);
}
else if (iNumberOfChannels ==2) {
cout << "Hue and Value" << endl;
//---------------------Back Projection---------------------
Mat hsv_roi;
Mat hist;
cvtColor( foreground, hsv_roi, COLOR_BGR2HSV );
int histSize = MAX( 100, 2 );
float h_range[] = { 0, 179 };
float s_range[] = { 0, 255 };
const float* ranges[] = { h_range, s_range };
int channels[] = { 0, 1 };
/// Get the Histogram and normalize it
calcHist( &hsv_roi, 1, channels, maskGrabcut, hist, 2, &histSize, ranges, true, false );
normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );
//
Mat hsv_target;
cvtColor( imageKinect, hsv_target, COLOR_BGR2HSV );
calcBackProject( &hsv_target, 1, channels, hist, backproj, ranges, 1, true );
//imshow("Back Projection", backproj);
}
else {
cout << "No valid value for iNumberOfChannels" << endl;
}
bool bFilter = true;
Mat mask;
if(bFilter) {
threshold(backproj, mask, 0, 255, THRESH_BINARY);
int morph_size1 = 3;
Mat element1 = getStructuringElement( 0, Size( 2*morph_size1 + 1, 2*morph_size1+1 ), Point( morph_size1, morph_size1 ) );
morphologyEx(mask, mask, MORPH_OPEN, element1);
int morph_size2 = 5;
Mat element2 = getStructuringElement( 0, Size( 2*morph_size2 + 1, 2*morph_size2+1 ), Point( morph_size2, morph_size2 ) );
morphologyEx(mask, mask, MORPH_CLOSE, element2);
morphologyEx(mask, mask, MORPH_DILATE, element2, Point(), 2);
Mat backprojection;
backproj.copyTo(backprojection, mask);
backprojection.copyTo(backproj);
//imshow("Filtered backproj", backproj);
}
backproj.copyTo(imageProbabilityMap);
cv::normalize(imageProbabilityMap, imageProbabilityMap, 0, 255, NORM_MINMAX, CV_8UC1);
}
}
std::vector< DMatch > CreateProbabilityMapGaze::keypointsHD() {
int iMinHessian = 400; //minimum Hessian threshold, 300~500
//Surf Feature Detector and Extractor
SurfDescriptorExtractor extractor;
SurfFeatureDetector detector( iMinHessian );
Mat descriptors_object, descriptors_scene;
FlannBasedMatcher matcher;
std::vector< DMatch > matches;
//Compute Keypoints in Scene image
detector.detect( imageHD, keypoints_scene );
extractor.compute( imageHD, keypoints_scene, descriptors_scene );
//-- Step 1: Get subimage with object
Mat object = imageEye(getROI(gaze_direction_eyetracker,iWindowSize));
//-- Step 2: Detect the keypoints using SURF Detector
detector.detect( object, keypoints_object );
//-- Step 3: Calculate descriptors (feature vectors)
extractor.compute( object, keypoints_object, descriptors_object );
//-- Step 3: Matching descriptor vectors using FLANN matcher
matcher.match( descriptors_object, descriptors_scene, matches );
// -------------------Filter points based on distance quality-----------------------
double max_dist = 0;
double min_dist = 100;
std::vector< DMatch > filtered_matches;
//-- Quick calculation of max and min distances between keypoints
for( int i = 0; i < descriptors_object.rows; i++ ) {
double dist = matches[i].distance;
if( dist < min_dist )
min_dist = dist;
if( dist > max_dist )
max_dist = dist;
}
//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
std::vector< DMatch > good_matches;
for( int i = 0; i < descriptors_object.rows; i++ ){
if( matches[i].distance < 3*min_dist ) {
filtered_matches.push_back(matches[i]);
}
}
//-------------------------Plot---------------------------------------
cout << "Good Matches: " << filtered_matches.size() << endl;
//---------------------Transform from HD pixels to kinect pixels ------------------------------------
Eigen::MatrixXd M_extrinsic_eigen_HD_to_kinect, M_extrinsic_eigen_kinect_to_HD;
Eigen::MatrixXd A_kinect_eigen;
Eigen::MatrixXd A_hdcamera_eigen;
cv::cv2eigen(M_extrinsic_HD_to_kinect,M_extrinsic_eigen_HD_to_kinect);
cv::cv2eigen(M_extrinsic_kinect_to_HD,M_extrinsic_eigen_kinect_to_HD);
cv::cv2eigen(A_kinect,A_kinect_eigen);
cv::cv2eigen(A_hdcamera,A_hdcamera_eigen);
double dDepthObject = dMeanDepthObjects*1000;//in mm
std::vector<KeyPoint> keypoints_scene_kinect = keypoints_scene;
//Transform keypoints from HD camera pixel coordinates to Kinect RGB image pixel coordinates
for(int iNumberOfKeypoints=0; iNumberOfKeypoints < filtered_matches.size(); iNumberOfKeypoints++) {
//1). Transform HD to HD world camera coordinates using projection matrix HD camera
Eigen::Vector3d X_world_hd = A_hdcamera_eigen.inverse() * Eigen::Vector3d(keypoints_scene[filtered_matches[iNumberOfKeypoints].trainIdx].pt.x, keypoints_scene[filtered_matches[iNumberOfKeypoints].trainIdx].pt.y, 1.0) * dDepthObject;
//2). Transfrom HD world camera coordinates to kinect world camera coordinates using extrinsic stereo R,T
Eigen::Vector3d X_world_kinect = M_extrinsic_eigen_HD_to_kinect * Eigen::Vector4d(X_world_hd.x(), X_world_hd.y(),X_world_hd.z(), 1.0);
//3). Transform to kinect pixel coordinates using projection matrix kinect camera
Eigen::Vector3d X_pixel_kinect = A_kinect_eigen * X_world_kinect;
X_pixel_kinect = X_pixel_kinect/ X_pixel_kinect.z();
keypoints_scene_kinect.at(filtered_matches[iNumberOfKeypoints].trainIdx).pt = Point2f(X_pixel_kinect.x(), X_pixel_kinect.y());
}
//-------------------------------------------Creating mask-------------------------------------------------------
Mat Mask = Mat::zeros(imageKinect.size(), CV_8UC1);
Mat GaussianBlurredMask;
for (int i = 0; i < filtered_matches.size(); i++) {
circle(Mask, Point2f(keypoints_scene_kinect[filtered_matches[i].trainIdx].pt.x, keypoints_scene_kinect[filtered_matches[i].trainIdx].pt.y), 1, Scalar(255,255,255), -1, 8 );
}
GaussianBlur( Mask, GaussianBlurredMask, Size( iSizeGaussianBlurFilter, iSizeGaussianBlurFilter), 0, 0 );
cv::normalize(GaussianBlurredMask, GaussianBlurredMask, 0, 255, NORM_MINMAX, CV_8UC1);
// The probability map is computed as a product of the last probability maps Added 6-1-2015
if(iNumberOfMaps>1) {
maps.erase(maps.begin());
maps.push_back(GaussianBlurredMask);
Mat combined;
for(int i = 0; i<maps.size(); i++) {
if(i==0) {
combined = maps[i];
}
else {
Mat image_32FC1 = maps[i].clone();
image_32FC1.convertTo(image_32FC1, CV_32FC1);
Mat combined_32FC1 = combined.clone();
combined_32FC1.convertTo(combined_32FC1, CV_32FC1);
Mat product = image_32FC1+combined_32FC1;
product.convertTo(combined, CV_32FC1, 1.0f / (255.0f+255.0f) * 255);
combined.convertTo(combined, CV_8UC1);
}
}
cv::normalize(combined, combined, 0, 255, NORM_MINMAX, CV_8UC1);
combined.copyTo(imageProbabilityMap);
}
else {
GaussianBlurredMask.copyTo(imageProbabilityMap);
}
//Show keypoint matches
if (b2Dviewer) {
Mat img_matches_kinect;
drawMatches(object, keypoints_object, imageKinect,
keypoints_scene_kinect, filtered_matches, img_matches_kinect, Scalar::all(-1),
Scalar::all(-1), vector<char>(),
DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
imshow("Good matches feature points in Kinect frame", img_matches_kinect);
drawMatches(object, keypoints_object, imageHD,
keypoints_scene, filtered_matches, img_matches, Scalar::all(-1),
Scalar::all(-1), vector<char>(),
DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
namedWindow("Good matches feature points", CV_WINDOW_FREERATIO );
imshow("Good matches feature points", img_matches);
resizeWindow("Good matches feature points", 640, 480);
}
return filtered_matches;
}
//========================================================================================================================
std::vector< DMatch > CreateProbabilityMapGaze::keypointsKinect() {
int iMinHessian = 400; //minimum Hessian threshold, 300~500
//Surf Feature Detector and Extractor
SurfDescriptorExtractor extractor;
SurfFeatureDetector detector( iMinHessian );
Mat descriptors_object, descriptors_scene;
FlannBasedMatcher matcher;
std::vector< DMatch > matches;
//Compute Keypoints in Scene image
detector.detect( imageKinect, keypoints_scene );
extractor.compute( imageKinect, keypoints_scene, descriptors_scene );
//-- Step 1: Get subimage with object
Mat object = imageEye(getROI(gaze_direction_eyetracker,iWindowSize));
//-- Step 2: Detect the keypoints using SURF Detector
detector.detect( object, keypoints_object );
//-- Step 3: Calculate descriptors (feature vectors)
extractor.compute( object, keypoints_object, descriptors_object );
//-- Step 3: Matching descriptor vectors using FLANN matcher
matcher.match( descriptors_object, descriptors_scene, matches );
// -------------------Filter points based on distance quality-----------------------
double max_dist = 0;
double min_dist = 100;
std::vector< DMatch > filtered_matches;
//-- Quick calculation of max and min distances between keypoints
for( int i = 0; i < descriptors_object.rows; i++ ) {
double dist = matches[i].distance;
if( dist < min_dist )
min_dist = dist;
if( dist > max_dist )
max_dist = dist;
}
//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
std::vector< DMatch > good_matches;
for( int i = 0; i < descriptors_object.rows; i++ ){
if( matches[i].distance < 3*min_dist ) {
filtered_matches.push_back(matches[i]);
}
}
//-------------------------Plot---------------------------------------
cout << "Good Matches: " << filtered_matches.size() << endl;
//-------------------------------------------Creating mask-------------------------------------------------------
Mat Mask = Mat::zeros(imageKinect.size(), CV_8UC1);
Mat GaussianBlurredMask;
for (int i = 0; i < filtered_matches.size(); i++) {
circle(Mask, Point2f(keypoints_scene[filtered_matches[i].trainIdx].pt.x, keypoints_scene[filtered_matches[i].trainIdx].pt.y), 1, Scalar(255,255,255), -1, 8 );
}
GaussianBlur( Mask, GaussianBlurredMask, Size( iSizeGaussianBlurFilter, iSizeGaussianBlurFilter), 0, 0 );
cv::normalize(GaussianBlurredMask, GaussianBlurredMask, 0, 255, NORM_MINMAX, CV_8UC1);
// The probability map is computed as a product of the last probability maps Added 6-1-2015
if(iNumberOfMaps>1) {
maps.erase(maps.begin());
maps.push_back(GaussianBlurredMask);
Mat combined;
for(int i = 0; i<maps.size(); i++) {
if(i==0) {
combined = maps[i];
}
else {
Mat image_32FC1 = maps[i].clone();
image_32FC1.convertTo(image_32FC1, CV_32FC1);
Mat combined_32FC1 = combined.clone();
combined_32FC1.convertTo(combined_32FC1, CV_32FC1);
Mat product = image_32FC1+combined_32FC1;
product.convertTo(combined, CV_32FC1, 1.0f / (255.0f+255.0f) * 255);
combined.convertTo(combined, CV_8UC1);
}
}
cv::normalize(combined, combined, 0, 255, NORM_MINMAX, CV_8UC1);
combined.copyTo(imageProbabilityMap);
}
else {
GaussianBlurredMask.copyTo(imageProbabilityMap);
}
//Show keypoints
if (b2Dviewer) {
drawMatches(object, keypoints_object, imageKinect,
keypoints_scene, filtered_matches, img_matches, Scalar::all(-1),
Scalar::all(-1), vector<char>(),
DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
namedWindow("Good matches feature points", CV_WINDOW_FREERATIO );
imshow("Good matches feature points", img_matches);
resizeWindow("Good matches feature points", 640, 480);
}
return filtered_matches;
}
Rect CreateProbabilityMapGaze::getROI(cv::Point Center, int iSizeWindow) {
int iRectangle_x = Center.x - iSizeWindow/2;
int iRectangle_y = Center.y - iSizeWindow/2;
int iRectangle_width = iSizeWindow;
int iRectangle_height = iSizeWindow;
if (iRectangle_x < 0)
iRectangle_x = 0;
if (iRectangle_y < 0)
iRectangle_y = 0;
if (iRectangle_x + iRectangle_width > cv_ptr_usb->image.cols)
iRectangle_width = cv_ptr_usb->image.cols -iRectangle_x;
if (iRectangle_y + iRectangle_height > cv_ptr_usb->image.rows)
iRectangle_height = cv_ptr_usb->image.rows -iRectangle_y;
return Rect(iRectangle_x,iRectangle_y,iRectangle_width,iRectangle_height);
}
Point2f CreateProbabilityMapGaze::getCenter(cv::Point Center, int iSizeWindow) {
int iRectangle_x = Center.x - iSizeWindow/2;
int iRectangle_y = Center.y - iSizeWindow/2;
int iRectangle_width = iSizeWindow;
int iRectangle_height = iSizeWindow;
if (iRectangle_x < 0)
iRectangle_x = 0;
if (iRectangle_y < 0)
iRectangle_y = 0;
if (iRectangle_x + iRectangle_width > cv_ptr_usb->image.cols)
iRectangle_width = cv_ptr_usb->image.cols -iRectangle_x;
if (iRectangle_y + iRectangle_height > cv_ptr_usb->image.rows)
iRectangle_height = cv_ptr_usb->image.rows -iRectangle_y;
return Point2f(iRectangle_x+ iRectangle_width/2, iRectangle_y+ iRectangle_height/2);
}
int main(int argc, char** argv)
{
ros::init(argc, argv, "Transform_Gaze");
CreateProbabilityMapGaze object;
ros::Rate rate(5); // 5Hz
while (ros::ok())
{
object.transformGaze();
rate.sleep();
ros::spinOnce();
}
return 0;
}

    Status
    API
    Training
    Shop
    Blog
    About

    © 2015 GitHub, Inc.
    Terms
    Privacy
    Security
    Contact

