#include <ros/ros.h>//Includes all the headers necessary to use the most common public pieces of the ROS system.
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <Eigen/Dense>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/imgproc/imgproc.hpp>//Include headers for OpenCV Image processing
#include <opencv2/highgui/highgui.hpp>//Include headers for OpenCV GUI handling
#include "opencv2/core/core.hpp"
#include "opencv/cv.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <image_transport/image_transport.h>//Use image_transport for publishing and subscribing to images in ROS
#include <geometry_msgs/PointStamped.h>
#include "gaze_msgs/PointingDirection.h"
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>//Use cv_bridge to convert between ROS and OpenCV Image formats
#include <sensor_msgs/image_encodings.h>//Include some useful constants for image encoding. Refer to: http://www.ros.org/doc/api/sensor_msgs/html/namespacesensor__msgs_1_1image__encodings.html for more info.

#include "pcl_ros/point_cloud.h"
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/impl/conditional_removal.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/time.h>
#include <boost/thread/thread.hpp>
#include <pcl/filters/passthrough.h>

#include <gaze_msgs/VectorPointcloudsClusters.h>

using namespace cv;
using namespace std;

#define PI 3.14159265
RNG rng(12345);

class IntegrateMaps {
protected:
public:
	ros::Subscriber sub_image_depth, sub_image_saliency, sub_image_pointing, sub_image_gaze;
	ros::Subscriber sub_image, sub_pcl, sub_camera_info, sub_pointing_dir, sub_markers, sub_clusters;
	ros::NodeHandle nh;

	image_transport::Publisher pubimage;

	Eigen::MatrixXf m;
	Point positionFinger;
	Point vectorFinger;
	Eigen::Vector3f positionFinger3D;
	Eigen::Vector3f vectorFinger3D;
	Mat saliencyMap;
	Mat depthMap;
	Mat pointingMap;
	Mat gazeMap;
	Mat imageKinect;
	int iAddOrMultiply;

	pcl::PointCloud<pcl::PointXYZ> cloud;
    visualization_msgs::Marker skelet;
    gaze_msgs::VectorPointcloudsClusters clusters;
    std::vector<pcl::PointCloud<pcl::PointXYZ> > vectorClusters;

	IntegrateMaps()
	{
		ros::NodeHandle private_nh("~");
		private_nh.param("addOrMultiply", iAddOrMultiply, 0);

		m.resize(3,4);
		cloud.width = 640; // Image-like organized structure, with 640 rows and 480 columns,
		cloud.height = 480; // thus 640*480=307200 points total in the dataset

		sub_image = nh.subscribe("/camera/rgb/image_raw", 1, &IntegrateMaps::callbackKinect, this);
		sub_image_depth = nh.subscribe("/segmenting/mask", 1, &IntegrateMaps::callbackDepth, this);
		sub_image_saliency = nh.subscribe("/saliency/image", 1, &IntegrateMaps::callbackSaliency, this);
		sub_image_pointing = nh.subscribe("/pointing/probabilitymap", 1, &IntegrateMaps::callbackPointing, this);
		sub_image_gaze = nh.subscribe("/gaze/probabilitymap", 1, &IntegrateMaps::callbackGaze, this);
		sub_pcl = nh.subscribe("/segmenting/cloud", 1, &IntegrateMaps::callbackPCL, this);
		sub_camera_info = nh.subscribe("/camera/rgb/camera_info", 1, &IntegrateMaps::callbackCameraInfo, this);
		sub_pointing_dir = nh.subscribe("/pointing/direction", 1, &IntegrateMaps::callbackPointingFinger, this);
		sub_markers = nh.subscribe("/skeleton_markers", 1, &IntegrateMaps::callbackSkelet, this);
		sub_clusters = nh.subscribe("/segmenting/clusters", 1, &IntegrateMaps::callbackClusters, this);
	}

	void callbackDepth(const sensor_msgs::ImageConstPtr& image1);
	void callbackSaliency(const sensor_msgs::ImageConstPtr& image1);
	void callbackPointing(const sensor_msgs::ImageConstPtr& image1);
	void callbackGaze(const sensor_msgs::ImageConstPtr& image1);
	void callbackKinect(const sensor_msgs::ImageConstPtr& image1);
	void callbackPCL(const sensor_msgs::PointCloud2ConstPtr& cloud_in);
	void callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& info);
	void callbackPointingFinger(const gaze_msgs::PointingDirectionConstPtr& pointingDir);
	void callbackSkelet(const visualization_msgs::MarkerConstPtr& markers);
	void callbackClusters(const gaze_msgs::VectorPointcloudsClustersConstPtr& clusters);

	void mainLoop();
	Mat combineMaps(std::vector<Mat> vectorWithImages);
	Point findMaximumLocation(Mat finalSaliencyMap);
	Mat segmentObject(std::vector<pcl::PointCloud<pcl::PointXYZ> > vectorClusters, Point maxLocation);
	pcl::PointCloud<pcl::PointXYZ> removeBody();
	Rect getROI(cv::Point Center, int iSizeWindow);
	void semgent2dMask(Point maxLocation);
};
void IntegrateMaps::callbackClusters(const gaze_msgs::VectorPointcloudsClustersConstPtr& clustersIN) {
	vectorClusters.clear();
	for (int i = 0; i<(*clustersIN).pointClouds.size(); i++) {
		pcl::PointCloud<pcl::PointXYZ> cluster;
		pcl::PCLPointCloud2 pcl_pc;
		pcl_conversions::toPCL((*clustersIN).pointClouds[i], pcl_pc);
		pcl::fromPCLPointCloud2(pcl_pc, cluster);
		vectorClusters.push_back(cluster);
	}
}
void IntegrateMaps::callbackSkelet(const visualization_msgs::MarkerConstPtr& markers) {
	skelet = *markers;
}
void IntegrateMaps::callbackPointingFinger(const gaze_msgs::PointingDirectionConstPtr& pointingDir) {
	gaze_msgs::PointingDirection in = (*pointingDir);
	positionFinger  = Point(in.position_2d.x, in.position_2d.y);
	vectorFinger  = Point(in.vector_2d.x, in.vector_2d.y);

	positionFinger3D << in.position_3d.x, in.position_3d.y, in.position_3d.z;
	vectorFinger3D << in.vector_3d.x, in.vector_3d.y, in.vector_3d.z;
}
void IntegrateMaps::callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& info) {
	sensor_msgs::CameraInfo cameraInfo = *info;
	m << cameraInfo.P[0],cameraInfo.P[1],cameraInfo.P[2],cameraInfo.P[3],cameraInfo.P[4],cameraInfo.P[5],cameraInfo.P[6],cameraInfo.P[7],cameraInfo.P[8],cameraInfo.P[9],cameraInfo.P[10],cameraInfo.P[11];
}
void IntegrateMaps::callbackPCL(const sensor_msgs::PointCloud2ConstPtr& cloud_in) {
	pcl::PCLPointCloud2 pcl_pc;
	pcl_conversions::toPCL(*cloud_in, pcl_pc);
	pcl::fromPCLPointCloud2(pcl_pc, cloud);
}
void IntegrateMaps::callbackKinect(const sensor_msgs::ImageConstPtr& image1) {
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
	cv_ptr->image.copyTo(imageKinect);
}
void IntegrateMaps::callbackDepth(const sensor_msgs::ImageConstPtr& image1) {
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
void IntegrateMaps::callbackSaliency(const sensor_msgs::ImageConstPtr& image1) {
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
	cv_ptr->image.copyTo(saliencyMap);
}
void IntegrateMaps::callbackPointing(const sensor_msgs::ImageConstPtr& image1) {
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
	cv_ptr->image.copyTo(pointingMap);
}
void IntegrateMaps::callbackGaze(const sensor_msgs::ImageConstPtr& image1) {
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
	cv_ptr->image.copyTo(gazeMap);
}
Mat IntegrateMaps::combineMaps(std::vector<Mat> vectorWithImages) {
	Mat combined;
	cout << "Number of available cues: " << vectorWithImages.size() << endl;
	for (int iNumberOfImages = 0; iNumberOfImages<vectorWithImages.size(); iNumberOfImages++) {
		if(iNumberOfImages == 1) {
			combined = vectorWithImages[iNumberOfImages];
		}
		else if(iNumberOfImages > 1) {
			if(iAddOrMultiply==0) {
				Mat image_32FC1 = vectorWithImages[iNumberOfImages].clone();
				image_32FC1.convertTo(image_32FC1, CV_32FC1);
				Mat combined_32FC1 = combined.clone();
				combined_32FC1.convertTo(combined_32FC1, CV_32FC1);

				Mat product = image_32FC1.mul(combined_32FC1);
				product.convertTo(combined, CV_32FC1, 1.0f / 65025.0f * 255);
				combined.convertTo(combined, CV_8UC1);
			}
			else if (iAddOrMultiply==1) {
				Mat image_32FC1 = vectorWithImages[iNumberOfImages].clone();
				image_32FC1.convertTo(image_32FC1, CV_32FC1);
				Mat combined_32FC1 = combined.clone();
				combined_32FC1.convertTo(combined_32FC1, CV_32FC1);

				Mat product = image_32FC1+combined_32FC1;
				product.convertTo(combined, CV_32FC1, 1.0f / (255.0f+255.0f) * 255);
				combined.convertTo(combined, CV_8UC1);
			}
			else {
				ROS_ERROR("Choose between add or multiply");
			}
		}
	}
	return combined;
}
Point IntegrateMaps::findMaximumLocation(Mat finalSaliencyMap) {
	double maxVal;
	Point maxLoc;
	Mat greyCombined;
	cvtColor(finalSaliencyMap, greyCombined, CV_RGB2GRAY);
	cv::minMaxLoc(greyCombined, NULL, &maxVal, NULL, &maxLoc, Mat());

	return maxLoc;
}
Mat IntegrateMaps::segmentObject(std::vector<pcl::PointCloud<pcl::PointXYZ> > vectorClusters, Point maxLocation) {
	Eigen::Vector4f centroid;
	std::vector<Point> centroidClusters;
	for (int i = 0;  i < vectorClusters.size(); i++) {
		//1). Compute centroid
		pcl::compute3DCentroid(vectorClusters[i], centroid);
		//2). Transform to 2D
		Eigen::VectorXf centroidCluster_3d(4); centroidCluster_3d<< centroid[0], centroid[1], centroid[2], 1.0;
		Eigen::Vector3f centroidCluster_2d = m*centroidCluster_3d;  centroidCluster_2d = centroidCluster_2d/centroidCluster_2d.z();
		centroidClusters.push_back(Point(centroidCluster_2d.x(), centroidCluster_2d.y()));
	}

	//3). find closest to maxLoc
	double dMinDistance = 100000;
	int iBestCluster = 0;
	for(int iNumberOfClusters = 0; iNumberOfClusters < vectorClusters.size(); iNumberOfClusters++) {
		double distance = (centroidClusters[iNumberOfClusters].x - maxLocation.x)*(centroidClusters[iNumberOfClusters].x - maxLocation.x) +  (centroidClusters[iNumberOfClusters].y - maxLocation.y)*(centroidClusters[iNumberOfClusters].y - maxLocation.y);
		if (distance < dMinDistance) {
			dMinDistance = distance;
			iBestCluster = iNumberOfClusters;
		}
	}

	//4). Compute mask with object.
	Mat maskWithObject = Mat::zeros(imageKinect.rows,imageKinect.cols, CV_8UC1);;
	for (int iNumberOfIndices = 0; iNumberOfIndices < vectorClusters[iBestCluster].points.size(); iNumberOfIndices++) {
		//Transform 3D point to 2D
		Eigen::VectorXf coordinates3d(4);
		coordinates3d << vectorClusters[iBestCluster].points[iNumberOfIndices].x, vectorClusters[iBestCluster].points[iNumberOfIndices].y, vectorClusters[iBestCluster].points[iNumberOfIndices].z, 1.0;
		Eigen::Vector3f coordinates2d = m*coordinates3d;  coordinates2d = coordinates2d/coordinates2d.z();

		//Mask containing all objects
		if (!isnan(coordinates2d.x()) && !isnan(coordinates2d.y())) {
			maskWithObject.at<uchar>(Point(coordinates2d.x(), coordinates2d.y())) = 255;
		}
	}

	//Close mask to remove gaps
	int morph_size = 3;
	Mat element = getStructuringElement( 0, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
	morphologyEx(maskWithObject, maskWithObject, MORPH_CLOSE, element, Point(-1,-1), 5);

	return maskWithObject;
}
void IntegrateMaps::mainLoop() {
	std::vector<Mat> vectorWithImages;
	if (!saliencyMap.empty()) { vectorWithImages.push_back(saliencyMap); }
	if (!depthMap.empty()) { vectorWithImages.push_back(depthMap); }
	if (!pointingMap.empty()) { vectorWithImages.push_back(pointingMap); }
	if (!gazeMap.empty()) {	vectorWithImages.push_back(gazeMap); }

	Mat finalSaliencyMap = combineMaps(vectorWithImages);

	Point maxLocation;
	Mat imageKinectWithCoordinates = imageKinect.clone();
	Mat image;
	if (!finalSaliencyMap.empty()) {
		maxLocation = findMaximumLocation(finalSaliencyMap);

		//Show
		circle(imageKinectWithCoordinates, maxLocation, 5, Scalar(0,0,255), -1);
		hconcat(imageKinectWithCoordinates, finalSaliencyMap, image);
	}

	//-----------------------------------------------------
	if (vectorClusters.size() > 0 && maxLocation!=Point(0,0)) {
		Mat mask = segmentObject(vectorClusters, maxLocation);
		cv::Mat object(imageKinect.size(),CV_8UC3,cv::Scalar(255,255,255));
		imageKinect.copyTo(object, mask);

		//Show
		hconcat(image, object, image);
		namedWindow("Image with object", CV_WINDOW_FREERATIO );
		imshow("Image with object", image);
		resizeWindow("Image with object", 1920, 480);
	}

	Mat allAvailableCues;
	if (!finalSaliencyMap.empty()) {
		for(int i =0; i<vectorWithImages.size(); i++) {
			if(i == 0) {
				allAvailableCues = vectorWithImages[0];
			}
			else {
				hconcat(vectorWithImages[i], allAvailableCues, allAvailableCues);
			}
		}
		namedWindow("Image with all available cues", CV_WINDOW_FREERATIO );
		imshow("Image with all available cues", allAvailableCues);
	}
	waitKey(3);
}
void IntegrateMaps::semgent2dMask(Point maxLocation) {
	if (!depthMap.empty()){
		//imshow("depthMap", depthMap);

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		Mat depthMapC1;
		cvtColor(depthMap, depthMapC1, CV_RGB2GRAY);
		findContours( depthMapC1, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

		/// Approximate contours to polygons + get bounding rects and circles
		vector<vector<Point> > contours_poly( contours.size() );
		vector<Rect> boundRect( contours.size() );
		vector<Point2f>center( contours.size() );
		vector<float>radius( contours.size() );

		for( int i = 0; i < contours.size(); i++ ){
			approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
			boundRect[i] = boundingRect( Mat(contours_poly[i]) );
			minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
		}

		/// Draw polygonal contour + bonding rects + circles
		Mat drawing = Mat::zeros( depthMapC1.size(), CV_8UC3 );
		for( int i = 0; i< contours.size(); i++ )
		{
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
			//drawContours( drawing, contours, (int)i, color, 2, 8, hierarchy, 0, Point() );
			rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
			circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
		}

		/// Show in a window
		//namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
		//imshow( "Contours", drawing );

		double dMinDistance = 1000;
		int iBestContour = 0;
		for(int iNumberOfContours = 0; iNumberOfContours < contours.size(); iNumberOfContours++) {
			double distance = (center[iNumberOfContours].x - maxLocation.x)*(center[iNumberOfContours].x - maxLocation.x) +  (center[iNumberOfContours].y - maxLocation.y)*(center[iNumberOfContours].y - maxLocation.y);
			if (distance < dMinDistance) {
				dMinDistance = distance;
				iBestContour = iNumberOfContours;
			}
		}
		Mat maskClosestContour = Mat::zeros( depthMapC1.size(), CV_8UC1 );
		rectangle( maskClosestContour, boundRect[iBestContour].tl(), boundRect[iBestContour].br(), Scalar(255), -1, 8, 0 );

		//namedWindow( "maskClosestContour", CV_WINDOW_AUTOSIZE );
		//imshow( "maskClosestContour", maskClosestContour );

		Mat maskWithObjectOfInterest;
		depthMap.copyTo(maskWithObjectOfInterest, maskClosestContour);

		//namedWindow( "maskWithObjectOfInterest", CV_WINDOW_AUTOSIZE );
		//imshow( "maskWithObjectOfInterest", maskWithObjectOfInterest );

		Mat object;
		imageKinect.copyTo(object, maskWithObjectOfInterest);
		imshow( "object 2d", object );
	}
}
Rect IntegrateMaps::getROI(cv::Point Center, int iSizeWindow) {

	int iRectangle_x = Center.x - iSizeWindow/2;
	int iRectangle_y = Center.y - iSizeWindow/2;
	int iRectangle_width = iSizeWindow;
	int iRectangle_height = iSizeWindow;

	if (iRectangle_x < 0)
		iRectangle_x = 0;
	if (iRectangle_y < 0)
		iRectangle_y = 0;
	if (iRectangle_x + iRectangle_width > imageKinect.cols)
		iRectangle_width = imageKinect.cols -iRectangle_x;
	if (iRectangle_y + iRectangle_height > imageKinect.rows)
		iRectangle_height = imageKinect.rows -iRectangle_y;

	return Rect(iRectangle_x,iRectangle_y,iRectangle_width,iRectangle_height);
}
pcl::PointCloud<pcl::PointXYZ> IntegrateMaps::removeBody() {
	Eigen::Vector3f rightElbow; rightElbow << -skelet.points[4].y, -skelet.points[4].z, skelet.points[4].x;
	Eigen::Vector3f rightHand; rightHand << -skelet.points[5].y, -skelet.points[5].z, skelet.points[5].x;
	Eigen::Vector3f leftElbow; leftElbow << -skelet.points[10].y, -skelet.points[10].z, skelet.points[10].x;
	Eigen::Vector3f leftHand; leftHand << -skelet.points[11].y, -skelet.points[11].z, skelet.points[11].x;
	Eigen::Vector3f limit;

	//Select correct elbow
	if((rightHand - positionFinger3D).norm() < (leftHand - positionFinger3D).norm()) {
		limit = (rightElbow + rightHand)/2;
	}
	else {
		limit = (leftElbow + leftHand)/2;
	}

	//---------------------Remove body -------------------------------
	pcl::PointCloud<pcl::PointXYZ> cloud_remove_body;
	cloud_remove_body.height = cloud.height;
	cloud_remove_body.width = cloud.width;

	pcl::PassThrough<pcl::PointXYZ> pass_remove_body;
	pass_remove_body.setInputCloud (cloud.makeShared());
	pass_remove_body.setFilterFieldName ("x");
	pass_remove_body.setKeepOrganized(true);
	//pass.setFilterLimitsNegative (true);
	if ( vectorFinger3D.x() > 0 ) {
		pass_remove_body.setFilterLimits (limit.x(), 10);
	}
	else {
		pass_remove_body.setFilterLimits (-10.0, limit.x());
	}
	pass_remove_body.filter (cloud_remove_body);

	return cloud_remove_body;
}
int main(int argc, char** argv)
{
	ros::init(argc, argv, "Create_pointing_map");

	IntegrateMaps object;
	ros::Rate rate(5); // 5Hz

	while (ros::ok())
	{
		object.mainLoop();
		rate.sleep();
		ros::spinOnce();
	}

	return 0;
}
