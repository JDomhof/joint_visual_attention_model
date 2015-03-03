#include <stdio.h>
#include <ros/ros.h>//Includes all the headers necessary to use the most common public pieces of the ROS system.
#include "pcl_ros/point_cloud.h"

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>

#include "skeleton_markers/Skeleton.h"

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl/conversions.h> //I believe you were using pcl/ros/conversion.h
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/impl/conditional_removal.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/time.h>
#include <boost/thread/thread.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/image_encodings.h>//Include some useful constants for image encoding. Refer to: http://www.ros.org/doc/api/sensor_msgs/html/namespacesensor__msgs_1_1image__encodings.html for more info.

#include <pcl/filters/passthrough.h>
#include <cv_bridge/cv_bridge.h>//Use cv_bridge to convert between ROS and OpenCV Image formats

#include <tf/transform_datatypes.h>
#include <Eigen/Dense>

#include <opencv2/imgproc/imgproc.hpp>//Include headers for OpenCV Image processing
#include <opencv2/highgui/highgui.hpp>//Include headers for OpenCV GUI handling
#include "opencv2/core/core.hpp"
#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include "gaze_msgs/PointingDirection.h"

#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/segmentation/extract_clusters.h>

#include <math.h>
#include <stdio.h>
#include <numeric>

using namespace std;
using namespace pcl;
using namespace cv;

void
displayEuclideanClusters (const pcl::PointCloud<pcl::PointXYZ>::CloudVectorType &clusters,
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
	char name[1024];
	unsigned char red [6] = {255,   0,   0, 255, 255,   0};
	unsigned char grn [6] = {  0, 255,   0, 255,   0, 255};
	unsigned char blu [6] = {  0,   0, 255,   0, 255, 255};

	for (size_t i = 0; i < clusters.size (); i++)
	{
		sprintf (name, "cluster_%d" , int (i));
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color0(boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(clusters[i]),red[i%6],grn[i%6],blu[i%6]);
		if (!viewer->updatePointCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(clusters[i]),color0,name))
			viewer->addPointCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(clusters[i]),color0,name);
		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, name);
		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, name);
	}
}
class Hand {
private:
protected:
	boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_, vis_cone_;
public:
	ros::Subscriber sub_markers, sub_pcl, sub_camera_info, sub_rgb_image;
	ros::Publisher pub_pointing_dir_;

	visualization_msgs::Marker skelet;
	pcl::PointCloud<pcl::PointXYZ> cloud;
	sensor_msgs::CameraInfo cameraInfo;
	sensor_msgs::PointCloud2 cloudMsg;
	cv_bridge::CvImagePtr cv_ptr_kinect;

	int iHand, iElbow;
	int iNumberOfMessages;
	double dRadiusCylinder;
	double dInitialHandSize;
	double dHandPalmRatio;
	bool bPCLviewer;
	bool b2Dviewer;
	std::string sHand;
	Mat imageKinect;
	Mat imageKinectWithCoordinates;

	Eigen::MatrixXf m;
	std::vector<Eigen::Vector3f> pointingDirection;
	std::vector<Eigen::Vector3f> pointingDirection2D;
	std::vector<Eigen::Vector3f> previous_hand_positions;

	pcl::PointCloud<pcl::PointXYZ> coneCloud;
	pcl::PointCloud<pcl::PointXYZ>::CloudVectorType prev_clusters_;

	Hand();

	void callbackPCL(const sensor_msgs::PointCloud2ConstPtr& cloud_in);
	void callbackSkelet(const visualization_msgs::MarkerConstPtr& markers);
	void callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& info);
	void callbackRGBImage(const sensor_msgs::ImageConstPtr& image1);

	void mainLoopPointingDirection();
	std::vector<Eigen::Vector3f> getAndShowPointingDirection2D();
	void publish();

	Eigen::Matrix3f getRotation(tf::Vector3 v1, tf::Vector3 v2);
	pcl::PointCloud<pcl::PointXYZ> getCone(pcl::PointCloud<pcl::PointXYZ> cloud, Eigen::Matrix3f R, Eigen::Vector3f T);
	pcl::PointCloud<pcl::PointXYZ> getCylinder(double radius, pcl::PointCloud<pcl::PointXYZ> cloud, Eigen::Matrix3f R, Eigen::Vector3f T);
	pcl::PointCloud<pcl::PointXYZ> getHand(double radius, pcl::PointCloud<pcl::PointXYZ> cloud, Eigen::Vector3f T);
	pcl::PointCloud<pcl::PointXYZ> filterPalm(double radius, pcl::PointCloud<pcl::PointXYZ> cloud, Eigen::Vector3f T);
	std::vector<pcl::PointIndices>  extractClusters(pcl::PointCloud<pcl::PointXYZ> cloud);
	std::vector<pcl::PointIndices>  extractClustersHand(pcl::PointCloud<pcl::PointXYZ> cloud);
	std::vector<Eigen::Vector4f> getCentroidsClusters(std::vector<pcl::PointIndices> hand_cluster_indices, pcl::PointCloud<pcl::PointXYZ> fingers);
	int getCentroidPointingFinger(Eigen::Vector3f elbow, std::vector<Eigen::Vector4f> centroid_clusters);
	pcl::PointCloud<pcl::PointXYZ> getCloudPointingFinger(int i, std::vector<pcl::PointIndices> hand_cluster_indices, pcl::PointCloud<pcl::PointXYZ> fingers);
	std::vector<Eigen::Vector3f> getPointingDirection(int iPointingFingerLeft, std::vector<Eigen::Vector4f> vector_centroids_clusters,  pcl::PointCloud<pcl::PointXYZ> pointingFingerLeft, Eigen::Vector3f refPoint);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> createVis();
};
Hand::Hand() {
	ros::NodeHandle n;
	ros::NodeHandle private_nh("~");

	private_nh.getParam("hand", sHand);
	private_nh.param("radiusCone", dRadiusCylinder, 0.25);
	private_nh.param("handSize", dInitialHandSize, 0.15);
	private_nh.param("handPalmRatio", dHandPalmRatio, 1.5);
	private_nh.param("view3D", bPCLviewer, false);
	private_nh.param("TwoDviewer", b2Dviewer, false);
	private_nh.param("numberOfMessages", iNumberOfMessages, 1);

	//======================Determine Left or Right hand==================
	if(sHand.compare("L")==0) {
		iHand = 11;	iElbow = 10;
	}
	else if (sHand.compare("R")==0) {
		iHand = 5; iElbow = 4;
	}
	else {
		cout << "Chose between L or R (Left or Right) Hand" << endl;
	}
	//=====================================================================
	if (bPCLviewer) {
		vis_ = createVis();
	}

	previous_hand_positions = std::vector<Eigen::Vector3f>(iNumberOfMessages, Eigen::Vector3f(0,0,0) );

	pub_pointing_dir_ = n.advertise <gaze_msgs::PointingDirection> ("/pointing/direction", 1);

	sub_markers = n.subscribe("/skeleton_markers", 1, &Hand::callbackSkelet, this);
	sub_pcl = n.subscribe("/camera/depth_registered/points", 1, &Hand::callbackPCL, this);
	sub_camera_info = n.subscribe("/camera/rgb/camera_info", 1, &Hand::callbackCameraInfo, this);
	sub_rgb_image = n.subscribe("/camera/rgb/image_raw", 1, &Hand::callbackRGBImage, this);
}
void Hand::callbackPCL(const sensor_msgs::PointCloud2ConstPtr& cloud_in) {
	pcl::PCLPointCloud2 pcl_pc;
	pcl_conversions::toPCL(*cloud_in, pcl_pc);
	pcl::fromPCLPointCloud2(pcl_pc, cloud);
	cloudMsg = *cloud_in;
}
void Hand::callbackSkelet(const visualization_msgs::MarkerConstPtr& markers) {
	skelet = *markers;
}
void Hand::callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& info) {
	cameraInfo = *info;
}
void Hand::callbackRGBImage(const sensor_msgs::ImageConstPtr& image1) {
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
boost::shared_ptr<pcl::visualization::PCLVisualizer> Hand::createVis() {
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

	viewer->setBackgroundColor (0, 0, 0);
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();
	viewer->setCameraPosition (0, 0, -1.5, 0, -0.5, 3, 0, -1, -0);
	return viewer;
}
std::vector<Eigen::Vector3f> Hand::getAndShowPointingDirection2D() {
	m.resize(3,4);
	m << cameraInfo.P[0],cameraInfo.P[1],cameraInfo.P[2],cameraInfo.P[3],cameraInfo.P[4],cameraInfo.P[5],cameraInfo.P[6],cameraInfo.P[7],cameraInfo.P[8],cameraInfo.P[9],cameraInfo.P[10],cameraInfo.P[11];
	std::vector<Eigen::Vector3f> temp;

	Eigen::VectorXf p1_finger_3d(4); p1_finger_3d << pointingDirection[0][0], pointingDirection[0][1], pointingDirection[0][2], 1.0;
	Eigen::Vector3f p1_finger_2d = m*p1_finger_3d;  p1_finger_2d = p1_finger_2d/p1_finger_2d.z();
	temp.push_back(p1_finger_2d);
	circle(imageKinectWithCoordinates, cv::Point2f(p1_finger_2d.x(), p1_finger_2d.y()), 8, Scalar(255,0,0), -1, 8 );

	double dLengthVector = 100;
	Eigen::Vector3f direction; direction << pointingDirection[1][0], pointingDirection[1][1], pointingDirection[1][2];
	Eigen::Vector3f position; position << pointingDirection[0][0], pointingDirection[0][1], pointingDirection[0][2];
	Eigen::Vector3f line = position + dLengthVector*direction;
	Eigen::VectorXf p2_finger_3d(4); p2_finger_3d << line.x(), line.y(), line.z(), 1.0;
	//Eigen::VectorXf p2_finger_3d(4); p2_finger_3d << pointingDirection[0][0]+10*pointingDirection[1][0], pointingDirection[0][1]+10*pointingDirection[1][1], pointingDirection[0][2]+10*pointingDirection[1][2], 1.0;

	Eigen::Vector3f p2_finger_2d = m*p2_finger_3d;  p2_finger_2d = p2_finger_2d/p2_finger_2d.z();
	temp.push_back(p2_finger_2d);
	//circle(imageKinectWithCoordinates, cv::Point2f(p2_finger_2d.x(), p2_finger_2d.y()), 8, Scalar(0,255,0), -1, 8 );;
	cv::line(imageKinectWithCoordinates, cv::Point2f(p1_finger_2d.x(), p1_finger_2d.y()), cv::Point2f(p2_finger_2d.x(), p2_finger_2d.y()), Scalar(255,255,255), 1);

	//imshow("Pointing Direction in RGB image of the Kinect", imageKinect);
	//waitKey(3);
	return temp;
}
void Hand::publish() {
	gaze_msgs::PointingDirection out;
	//3D
	out.position_3d.x = pointingDirection[0][0];
	out.position_3d.y = pointingDirection[0][1];
	out.position_3d.z = pointingDirection[0][2];
	out.vector_3d.x = pointingDirection[1][0];
	out.vector_3d.y = pointingDirection[1][1];
	out.vector_3d.z = pointingDirection[1][2];
	//2d
	out.position_2d.x = pointingDirection2D[0][0];
	out.position_2d.y = pointingDirection2D[0][1];
	out.position_2d.z = pointingDirection2D[0][2];
	out.vector_2d.x = pointingDirection2D[1][0];
	out.vector_2d.y = pointingDirection2D[1][1];
	out.vector_2d.z = pointingDirection2D[1][2];
	pub_pointing_dir_.publish(out);
}
void Hand::mainLoopPointingDirection() {
	pcl::PointCloud<pcl::PointXYZ> plotPointingDirection;
	pcl::PointCloud<pcl::PointXYZ> plotCentroid;
	pcl::PointCloud<pcl::PointXYZ> fingers;
	pcl::PointCloud<pcl::PointXYZ> hand;

	clock_t start = clock();
	if(skelet.points.size()>0 && cloud.points.size()>0) {
		double dStampCloudMsg = (double)cloudMsg.header.stamp.nsec/1000000000;
		double dStampSkeletMsg = (double)(skelet.header.stamp.sec -  cloudMsg.header.stamp.sec) + (double)skelet.header.stamp.nsec/1000000000;

		if((dStampSkeletMsg - dStampCloudMsg) < 1.0){ //Only continue if time difference is smaller than 1.0seconds
			imageKinect.copyTo(imageKinectWithCoordinates);

			//Markers are in other framework thus transform them to depth_registered pointcloud reference frame
			Eigen::Vector3f T; T << -skelet.points[iHand].y, -skelet.points[iHand].z, skelet.points[iHand].x;
			Eigen::Vector3f Elbow; Elbow << -skelet.points[iElbow].y, -skelet.points[iElbow].z, skelet.points[iElbow].x;

			previous_hand_positions.erase(previous_hand_positions.begin());
			previous_hand_positions.push_back(T);

			if (previous_hand_positions.at(0) != Eigen::Vector3f(0,0,0)) {
				Eigen::Vector3f sum  = std::accumulate(previous_hand_positions.begin(), previous_hand_positions.end(), Eigen::Vector3f(0.0, 0.0, 0.0));
				Eigen::Vector3f meanHandPosition = sum/ previous_hand_positions.size(); //(sum.x / previous_hand_positions.size(), sum.y / previous_hand_positions.size(), sum.z / previous_hand_positions.size());

				m.resize(3,4);
				m << cameraInfo.P[0],cameraInfo.P[1],cameraInfo.P[2],cameraInfo.P[3],cameraInfo.P[4],cameraInfo.P[5],cameraInfo.P[6],cameraInfo.P[7],cameraInfo.P[8],cameraInfo.P[9],cameraInfo.P[10],cameraInfo.P[11];

				Eigen::VectorXf elbow2d(4); elbow2d << Elbow.x(), Elbow.y(), Elbow.z(), 1.0;
				Eigen::Vector3f p1_elbow2d = m*elbow2d;  p1_elbow2d = p1_elbow2d/p1_elbow2d.z();
				circle(imageKinectWithCoordinates, cv::Point2f(p1_elbow2d.x(), p1_elbow2d.y()), 8, Scalar(0,0,255), -1, 8 );

				Eigen::VectorXf hand2d(4); hand2d << meanHandPosition.x(), meanHandPosition.y(), meanHandPosition.z(), 1.0;
				Eigen::Vector3f p1_hand2d = m*hand2d;  p1_hand2d = p1_hand2d/p1_hand2d.z();
				circle(imageKinectWithCoordinates, cv::Point2f(p1_hand2d.x(), p1_hand2d.y()), 8, Scalar(0,0,255), -1, 8 );

				//======================Reduce size ==================================
				double Delta = 0.2;
				// ----------------------- Z axis ------------------------------------
				pcl::PointCloud<pcl::PointXYZ> cloud_filtered;
				pcl::PassThrough<pcl::PointXYZ> passZ;
				passZ.setInputCloud (cloud.makeShared());
				passZ.setFilterFieldName ("z");
				passZ.setFilterLimits (T.z() -Delta, T.z() + Delta);
				passZ.filter (cloud_filtered);
				// ----------------------- Z axis ------------------------------------
				pcl::PassThrough<pcl::PointXYZ> passY;
				passY.setInputCloud (cloud_filtered.makeShared());
				passY.setFilterFieldName ("y");
				passY.setFilterLimits (T.y() - Delta, T.y() + Delta);
				passY.filter (cloud_filtered);
				// ----------------------- Z axis ------------------------------------
				pcl::PassThrough<pcl::PointXYZ> passX;
				passX.setInputCloud (cloud_filtered.makeShared());
				passX.setFilterFieldName ("x");
				passX.setFilterLimits (T.x() - Delta, T.x() + Delta);
				passX.filter (cloud_filtered);

				// =======================Get centroid of the hand ============================
				pcl::PointCloud<pcl::PointXYZ> handComputeCentroid = getHand(dInitialHandSize, cloud_filtered, meanHandPosition);

				//=============================Compute Centroids=======================================
				Eigen::Vector4f centroid;
				if (handComputeCentroid.size() > 0) {
					pcl::compute3DCentroid(handComputeCentroid,centroid);
					pcl::PointXYZ pt_centroid = pcl::PointXYZ (centroid[0], centroid[1], centroid[2]);
					Eigen::Vector3f CentroidHand; CentroidHand<< centroid[0], centroid[1], centroid[2];

					//Plot centroid
					Eigen::VectorXf centroidHand2d(4); centroidHand2d << CentroidHand.x(), CentroidHand.y(), CentroidHand.z(), 1.0;
					Eigen::Vector3f p1_centroidHand2d = m*centroidHand2d;  p1_centroidHand2d = p1_centroidHand2d/p1_centroidHand2d.z();
					circle(imageKinectWithCoordinates, cv::Point2f(p1_centroidHand2d.x(), p1_centroidHand2d.y()), 8, Scalar(0,255,0), -1, 8 );

					// Use cylindrical filter to remove large part of the arm
					pcl::PointCloud<pcl::PointXYZ> cylinderCloud;
					tf::Vector3 cylinder_axis_neutral = tf::Vector3(0.0,0.0,1.0);
					tf::Vector3 cylinder_axis_target = tf::Vector3(
							T.x() - Elbow.x(),
							T.y() - Elbow.y(),
							T.z() - Elbow.z());
					cylinder_axis_target.normalize();
					//cout << "Vector pointing:" << cylinder_axis_target.x() << ", " << cylinder_axis_target.y() << ", " << cylinder_axis_target.z() << endl;

					Eigen::Vector3f tCylinder = Eigen::Vector3f(CentroidHand.x(), CentroidHand.y(), CentroidHand.z());
					Eigen::Matrix3f rCylinder = getRotation(cylinder_axis_neutral, cylinder_axis_target);
					cylinderCloud = getCylinder(dRadiusCylinder, cloud_filtered, rCylinder, tCylinder);

					//Extract clusters and find closest cluster(==hand)
					std::vector<pcl::PointIndices> filtered_hand_cluster_indices;
					filtered_hand_cluster_indices = extractClustersHand(cylinderCloud);
					//cout << "Number filtered hand cluster indices: " << filtered_hand_cluster_indices.size() << endl;

					pcl::PointCloud<pcl::PointXYZ>::CloudVectorType clusters;
					std::vector<Eigen::Vector3f> vectorCentroids;
					for (size_t i = 0; i < filtered_hand_cluster_indices.size (); i++)
					{
						if (filtered_hand_cluster_indices[i].indices.size () > 30)
						{
							pcl::PointCloud<pcl::PointXYZ> cluster;
							pcl::copyPointCloud (cylinderCloud,filtered_hand_cluster_indices[i].indices,cluster);
							clusters.push_back (cluster);

							//Compute centroid
							Eigen::Vector4f centroid;
							pcl::compute3DCentroid(cluster,centroid);

							Eigen::Vector3f coordinates3d(3);
							coordinates3d << centroid[0], centroid[1], centroid[2];
							vectorCentroids.push_back(coordinates3d);
						}
					}
					double minDistance = 1000;
					int iNumberHandCluster;
					for (int i = 0; i<vectorCentroids.size(); i++) {
						double distance = (vectorCentroids[i] - CentroidHand).norm();
						if(distance < minDistance) {
							minDistance = distance;
							iNumberHandCluster = i;
						}
					}
					pcl::copyPointCloud (clusters[iNumberHandCluster], hand);

					//Find centroid of new hand
					pcl::compute3DCentroid(clusters[iNumberHandCluster],centroid);
					Eigen::Vector3f centroid2; centroid2 << centroid[0], centroid[1], centroid[2];
					pcl::PointXYZ pt_centroid_handcluster = pcl::PointXYZ (centroid[0], centroid[1], centroid[2]);
					plotCentroid.push_back(pt_centroid_handcluster);

					//----------------------------------------------------------------------------------
					//TODO: tuning parameter
					double dRadiusPalm = dHandPalmRatio*(centroid2 - CentroidHand).norm();
					fingers = filterPalm(dRadiusPalm, clusters[iNumberHandCluster], CentroidHand);

					//=======================Extract Clusters ====================================
					std::vector<pcl::PointIndices> hand_cluster_indices;
					if (fingers.size() > 0) {
						hand_cluster_indices = extractClusters(fingers);
						if (hand_cluster_indices.size() > 0) {
							//==============================Find clusters with largest distance from elbow===========================
							std::vector<Eigen::Vector4f> vector_centroids_clusters = getCentroidsClusters(hand_cluster_indices, fingers);

							int iPointingFinger = getCentroidPointingFinger(Elbow, vector_centroids_clusters);

							pcl::PointCloud<pcl::PointXYZ> pointingFinger= getCloudPointingFinger(iPointingFinger, hand_cluster_indices, fingers);

							//==================================Compute eigenvector of finger========================================================
							pointingDirection = getPointingDirection(iPointingFinger, vector_centroids_clusters, pointingFinger, Elbow);

							pcl::PointXYZ p1 = pcl::PointXYZ(pointingDirection[0][0], pointingDirection[0][1], pointingDirection[0][2]);
							pcl::PointXYZ p2 = pcl::PointXYZ(pointingDirection[0][0]+pointingDirection[1][0], pointingDirection[0][1]+pointingDirection[1][1], pointingDirection[0][2]+pointingDirection[1][2]);

							plotPointingDirection.push_back(p1);
							plotPointingDirection.push_back(p2);
							plotPointingDirection.push_back(pt_centroid);

							pointingDirection2D = getAndShowPointingDirection2D();
							publish();
						}
						else {
							cout << "No clusters found" << endl;
						}
					}
					else {
						cout << "No fingers found(fingers pointcloud size = 0)" << endl;
					}
				}
				else {
					cout << "No hand found" << endl;
				}

				//----------------------Plot Results----------------------------
				//Plot 3D restuls only if true;
				if(bPCLviewer) {
					pcl::PointCloud<pcl::PointXYZ> plotHandlocation;
					pcl::PointXYZ pt_hand_position = pcl::PointXYZ ( meanHandPosition.x(),  meanHandPosition.y(),  meanHandPosition.z());
					plotHandlocation.push_back(pt_hand_position);

					pcl::visualization::PointCloudColorHandlerCustom <pcl::PointXYZ> color_hand_location (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(plotHandlocation), 255, 255, 0);
					if (!vis_->updatePointCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(plotHandlocation), color_hand_location,"Hand location")) {
						vis_->addPointCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(plotHandlocation), color_hand_location,"Hand location");
						vis_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "Hand location");
					}

					pcl::visualization::PointCloudColorHandlerCustom <pcl::PointXYZ> color_cloud_filtered (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(cloud_filtered), 255, 255, 255);
					if (!vis_->updatePointCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(cloud_filtered), color_cloud_filtered,"cloud_filtered")) {
						vis_->addPointCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(cloud_filtered), color_cloud_filtered,"cloud_filtered");
						vis_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 0.3, "cloud_filtered");
					}

					pcl::visualization::PointCloudColorHandlerCustom <pcl::PointXYZ> color_centroid_cluster (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(plotCentroid), 0, 255, 0);
					if (!vis_->updatePointCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(plotCentroid), color_centroid_cluster,"plotCentroid")) {
						vis_->addPointCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(plotCentroid), color_centroid_cluster,"plotCentroid");
						vis_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "plotCentroid");
					}

					pcl::visualization::PointCloudColorHandlerCustom <pcl::PointXYZ> color_hand (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(hand), 255, 0, 255);
					if (!vis_->updatePointCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(hand), color_hand,"hand")) {
						vis_->addPointCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(hand), color_hand,"hand");
						vis_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "hand");
						vis_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, "hand");
					}

					pcl::visualization::PointCloudColorHandlerCustom <pcl::PointXYZ> color_fingers (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(fingers), 255, 0, 0);
					if (!vis_->updatePointCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(fingers), color_fingers,"fingers")) {
						vis_->addPointCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(fingers), color_fingers,"fingers");
						vis_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "fingers");
						vis_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, "fingers");
					}

					pcl::visualization::PointCloudColorHandlerCustom <pcl::PointXYZ> color_centroid_fingers (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(plotPointingDirection), 0, 0, 255);
					if (!vis_->updatePointCloud (plotPointingDirection.makeShared(), color_centroid_fingers, "Centroid_fingers")) {
						vis_->addPointCloud (plotPointingDirection.makeShared(), color_centroid_fingers, "Centroid_fingers");
						vis_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "Centroid_fingers");
					}

					vis_->spinOnce();
				}
				//Plot image only if true
				if(b2Dviewer && (!imageKinectWithCoordinates.empty()) ) {
					imshow("Pointing Direction in RGB image of the Kinect", imageKinectWithCoordinates);
					waitKey(3);
				}
			}
		}
		else {
			ROS_ERROR("Messages unsynchronised. Time difference: %f", (dStampSkeletMsg - dStampCloudMsg));
		}
	}
	clock_t end = clock();
	cout << "Estimating pointing direction frame took: " << (float)(end - start) / CLOCKS_PER_SEC << " seconds" << endl;
}
pcl::PointCloud<pcl::PointXYZ> Hand::getCylinder(double radius, pcl::PointCloud<pcl::PointXYZ> cloud, Eigen::Matrix3f R, Eigen::Vector3f T) {
	pcl::PointCloud<pcl::PointXYZ> cloud_cone;

	//======================Start filtering================================
	// -------------------Condition ------------------------------------
	ConditionAnd<PointXYZ>::Ptr cyl_cond (new ConditionAnd<PointXYZ> ());

	//------------------------------------Cylinder-------------------------
	Eigen::Matrix3f sphereMatrix;
	Eigen::Matrix3f A; 	A << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0; //cylinder in z axis
	sphereMatrix = R.transpose() * A * R;
	Eigen::Vector3f sphereVector;
	sphereVector = -T.transpose()*sphereMatrix;
	float sphereScalar = -radius*radius + T.transpose() * sphereMatrix *  T; //radius² of sphere

	TfQuadraticXYZComparison<PointXYZ>::Ptr cyl_comp (new TfQuadraticXYZComparison<PointXYZ> (
			ComparisonOps::LE,
			sphereMatrix,
			sphereVector,
			sphereScalar));
	cyl_cond->addComparison (cyl_comp);

	//Condition for only postive z-direction
	Eigen::Vector3f vz_; vz_ << 0, 0, 0.5;
	Eigen::Vector3f zVector = vz_.transpose()*R.transpose();
	float zScalar = -2*vz_.transpose()*R.transpose()*T;
	TfQuadraticXYZComparison<PointXYZ>::Ptr z_comp (new TfQuadraticXYZComparison<PointXYZ> (
			ComparisonOps::GE,
			Eigen::Matrix3f::Zero(),
			zVector,
			zScalar));
	cyl_cond->addComparison (z_comp);

	//------------------- build the filter---------------------
	pcl::ConditionalRemoval<pcl::PointXYZ>condrem (cyl_cond);
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	condrem.setInputCloud (cloud.makeShared());
	condrem.setKeepOrganized(false);

	//---------------------apply filter-----------------------
	condrem.filter(cloud_cone);

	return cloud_cone;
}
pcl::PointCloud<pcl::PointXYZ> Hand::getCone(pcl::PointCloud<pcl::PointXYZ> cloud, Eigen::Matrix3f R, Eigen::Vector3f T) {
	pcl::PointCloud<pcl::PointXYZ> cloud_cone;

	//======================Start filtering================================
	// -------------------Condition ------------------------------------
	ConditionAnd<PointXYZ>::Ptr cyl_cond (new ConditionAnd<PointXYZ> ());

	//------------------------------------Cylinder-------------------------
	/*	Eigen::Matrix3f sphereMatrix;
	Eigen::Matrix3f A; 	A << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0; //cylinder in z axis
	sphereMatrix = R.transpose() * A * R;
	Eigen::Vector3f sphereVector;
	sphereVector = -T.transpose()*sphereMatrix;
	float sphereScalar = -0.15*0.15 + T.transpose() * sphereMatrix *  T; //radius² of sphere

	TfQuadraticXYZComparison<PointXYZ>::Ptr cyl_comp (new TfQuadraticXYZComparison<PointXYZ> (
			ComparisonOps::LE,
			sphereMatrix,
			sphereVector,
			sphereScalar));
	cyl_cond->addComparison (cyl_comp);*/

	//--------------------cone with radius r at h from origin------------------------
	//cout << "angle = " << dRadiusCone << ", " << tan(dRadiusCone * M_PI/180) << endl;

	double h = 1;	double r = h*tan(dRadiusCylinder * M_PI/180);		double c = r/h;		double z0 = 0;
	Eigen::Matrix3f A; 	A << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -c*c; //cylinder in z axis
	Eigen::Vector3f v_; v_ << 0, 0, 0;

	Eigen::Matrix3f sphereMatrix = R.transpose() * A * R;
	Eigen::Vector3f sphereVector = -T.transpose()*sphereMatrix + v_.transpose()*R.transpose();
	float sphereScalar = -z0*z0*c*c + T.transpose() * sphereMatrix *  T - 2*v_.transpose()*R.transpose()*T; //radius² of sphere

	TfQuadraticXYZComparison<PointXYZ>::Ptr cyl_comp (new TfQuadraticXYZComparison<PointXYZ> (
			ComparisonOps::LE,
			sphereMatrix,
			sphereVector,
			sphereScalar));
	cyl_cond->addComparison (cyl_comp);

	//Condition for only postive z-direction
	Eigen::Vector3f vz_; vz_ << 0, 0, 0.5;
	Eigen::Vector3f zVector = vz_.transpose()*R.transpose();
	float zScalar = -2*vz_.transpose()*R.transpose()*T;
	TfQuadraticXYZComparison<PointXYZ>::Ptr z_comp (new TfQuadraticXYZComparison<PointXYZ> (
			ComparisonOps::GE,
			Eigen::Matrix3f::Zero(),
			zVector,
			zScalar));
	cyl_cond->addComparison (z_comp);

	//------------------- build the filter---------------------
	pcl::ConditionalRemoval<pcl::PointXYZ>condrem (cyl_cond);
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	condrem.setInputCloud (cloud.makeShared());
	condrem.setKeepOrganized(false);

	//---------------------apply filter-----------------------
	condrem.filter(cloud_cone);

	return cloud_cone;
}
std::vector<Eigen::Vector3f> Hand::getPointingDirection(int iPointingFingerLeft, std::vector<Eigen::Vector4f> vector_centroids_clusters,  pcl::PointCloud<pcl::PointXYZ> pointingFingerLeft, Eigen::Vector3f referencePoint) {
	std::vector<Eigen::Vector3f> directionVector;

	Eigen::Matrix3f cov;
	EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
	EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
	Eigen::Vector3f centroid, direction;
	pcl::computeCovarianceMatrixNormalized(pointingFingerLeft,vector_centroids_clusters[iPointingFingerLeft],cov);
	pcl::eigen33 (cov, eigen_vectors, eigen_values);
	direction(0)=eigen_vectors (0, 2);
	direction(1)=eigen_vectors (1, 2);
	direction(2)=eigen_vectors (2, 2);

	Eigen::Vector3f centroid_pointing_finger_eigen = Eigen::Vector3f(vector_centroids_clusters[iPointingFingerLeft][0], vector_centroids_clusters[iPointingFingerLeft][1], vector_centroids_clusters[iPointingFingerLeft][2]);
	Eigen::Vector3f p1 = centroid_pointing_finger_eigen - direction;
	Eigen::Vector3f p2 = centroid_pointing_finger_eigen + direction; //from meter to cm

	directionVector.push_back(centroid_pointing_finger_eigen);
	//Compute distance
	double distance1 = (p1 - referencePoint).norm();
	double distance2 = (p2 - referencePoint).norm();
	if (distance1 > distance2) {
		//Eigen::Vector3f p = centroid_pointing_finger_eigen - 0.01 * direction;
		Eigen::Vector3f p =  - 0.01 * direction;
		directionVector.push_back(p);
	}
	else if (distance1 < distance2) {
		//Eigen::Vector3f p = centroid_pointing_finger_eigen + 0.01 * direction;
		Eigen::Vector3f p =  0.01 * direction;
		directionVector.push_back(p);
	}
	else {
		cout << "distance1 == distance2" << endl;
		//Eigen::Vector3f p = centroid_pointing_finger_eigen + 0.01 * direction;
		Eigen::Vector3f p = 0.01 * direction;
		directionVector.push_back(p);
	}

	return directionVector;
}
pcl::PointCloud<pcl::PointXYZ> Hand::getCloudPointingFinger(int i, std::vector<pcl::PointIndices> hand_cluster_indices, pcl::PointCloud<pcl::PointXYZ> fingers){

	pcl::PointCloud<pcl::PointXYZ> cloud_cluster;
	for (std::vector<int>::const_iterator pit = hand_cluster_indices[i].indices.begin (); pit != hand_cluster_indices[i].indices.end (); pit++) {
		cloud_cluster.points.push_back (fingers.points[*pit]); //*
	}
	cloud_cluster.width = cloud_cluster.points.size ();
	cloud_cluster.height = 1;
	cloud_cluster.is_dense = true;

	return cloud_cluster;
}
std::vector<Eigen::Vector4f> Hand::getCentroidsClusters(std::vector<pcl::PointIndices> hand_cluster_indices, pcl::PointCloud<pcl::PointXYZ> fingers){
	pcl::PointCloud<pcl::PointXYZ>::Ptr centroid_fingers_ptr (new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<Eigen::Vector4f> vector_centroids_clusters;
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;

	for (std::vector<pcl::PointIndices>::const_iterator it = hand_cluster_indices.begin (); it != hand_cluster_indices.end (); ++it)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++) {

			cloud_cluster->points.push_back (fingers.points[*pit]); //*
		}
		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		//Compute Centroid
		Eigen::Vector4f centroid;
		pcl::compute3DCentroid(*cloud_cluster,centroid);
		vector_centroids_clusters.push_back(Eigen::Vector4f(centroid[0], centroid[1], centroid[2], centroid[3]));
	}
	return vector_centroids_clusters;
}
int Hand::getCentroidPointingFinger(Eigen::Vector3f elbow, std::vector<Eigen::Vector4f> centroid_clusters) {
	double maxDistanceCentroidElbow = 0;
	int iPointingFinger;

	for(int i = 0; i<centroid_clusters.size(); i++) {
		double differenceX = (elbow[0] - centroid_clusters[i][0]);
		double differenceY = (elbow[1] - centroid_clusters[i][1]);
		double differenceZ = (elbow[2] - centroid_clusters[i][2]);
		double distanceCentroidElbow = differenceX*differenceX+differenceY*differenceY+differenceZ*differenceZ;
		if(distanceCentroidElbow > maxDistanceCentroidElbow) {
			maxDistanceCentroidElbow = distanceCentroidElbow;
			iPointingFinger = i;
		}
	}
	//cout << "Index of cluster of finger is: " << iPointingFinger << ", with distance: " << maxDistanceCentroidElbow << endl;
	return iPointingFinger;
}
std::vector<pcl::PointIndices>  Hand::extractClusters(pcl::PointCloud<pcl::PointXYZ> cloudCluster) {
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloudCluster.makeShared());

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (0.01); // 2cm
	ec.setMinClusterSize (50);
	ec.setMaxClusterSize (25000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (cloudCluster.makeShared());
	ec.extract (cluster_indices);
	return cluster_indices;
}
std::vector<pcl::PointIndices>  Hand::extractClustersHand(pcl::PointCloud<pcl::PointXYZ> cloudCluster) {
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloudCluster.makeShared());

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (0.02); // 2cm
	ec.setMinClusterSize (50);
	ec.setMaxClusterSize (25000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (cloudCluster.makeShared());
	ec.extract (cluster_indices);
	return cluster_indices;
}
//Filter Palm
pcl::PointCloud<pcl::PointXYZ> Hand::filterPalm(double radius, pcl::PointCloud<pcl::PointXYZ> cloud, Eigen::Vector3f T) {
	pcl::PointCloud<pcl::PointXYZ> cloud_hand;

	//======================Start filtering================================
	// -------------------Condition ------------------------------------
	ConditionAnd<PointXYZ>::Ptr cyl_cond (new ConditionAnd<PointXYZ> ());

	Eigen::Matrix3f sphereMatrix;
	sphereMatrix << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
	Eigen::Vector3f sphereVector;
	sphereVector = -T.transpose()*sphereMatrix;
	float sphereScalar = -radius*radius + T.transpose() * sphereMatrix *  T; //radius² of sphere

	TfQuadraticXYZComparison<PointXYZ>::Ptr cyl_comp (new TfQuadraticXYZComparison<PointXYZ> (ComparisonOps::GE, sphereMatrix,
			sphereVector, sphereScalar));
	cyl_cond->addComparison (cyl_comp);

	//------------------- build the filter---------------------
	pcl::ConditionalRemoval<pcl::PointXYZ>condrem (cyl_cond);
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	condrem.setInputCloud (cloud.makeShared());
	condrem.setKeepOrganized(false);

	//---------------------apply filter-----------------------
	condrem.filter(cloud_hand);

	return cloud_hand;
}
//GEt hand
pcl::PointCloud<pcl::PointXYZ> Hand::getHand(double radius, pcl::PointCloud<pcl::PointXYZ> cloud, Eigen::Vector3f T) {
	pcl::PointCloud<pcl::PointXYZ> cloud_hand;

	//======================Start filtering================================
	// -------------------Condition ------------------------------------
	ConditionAnd<PointXYZ>::Ptr cyl_cond (new ConditionAnd<PointXYZ> ());

	Eigen::Matrix3f sphereMatrix;
	sphereMatrix << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
	Eigen::Vector3f sphereVector;
	sphereVector = -T.transpose()*sphereMatrix;
	float sphereScalar = -radius*radius + T.transpose() * sphereMatrix *  T; //radius² of sphere

	TfQuadraticXYZComparison<PointXYZ>::Ptr cyl_comp (new TfQuadraticXYZComparison<PointXYZ> (ComparisonOps::LE, sphereMatrix,
			sphereVector, sphereScalar));
	cyl_cond->addComparison (cyl_comp);

	//------------------- build the filter---------------------
	pcl::ConditionalRemoval<pcl::PointXYZ>condrem (cyl_cond);
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	condrem.setInputCloud (cloud.makeShared());
	condrem.setKeepOrganized(false);

	//---------------------apply filter-----------------------
	condrem.filter(cloud_hand);

	return cloud_hand;
}
Eigen::Matrix3f Hand::getRotation(tf::Vector3 v1, tf::Vector3 v2) {
	//Normalize input vectors
	v1.normalize();
	v2.normalize();

	//Get rotation quaternion
	tf::Vector3 v3;
	v3 = v1.cross(v2);

	//Compute rotation quaternion
	tf::Quaternion q(v3.getX(), v3.getY(), v3.getZ(), 1 + v1.dot(v2));
	q.normalize();

	tf::Matrix3x3 m(q);
	Eigen::Matrix3f rotationMatrix;
	rotationMatrix << m.getRow(0)[0], m.getRow(0)[1], m.getRow(0)[2], m.getRow(1)[0], m.getRow(1)[1], m.getRow(1)[2], m.getRow(2)[0], m.getRow(2)[1], m.getRow(2)[2];

	return rotationMatrix;
}
int main(int argc, char** argv)
{
	ros::init(argc, argv, "Hand_node");

	Hand object;
	ros::Rate rate(5); // 5Hz

	while (ros::ok())
	{
		object.mainLoopPointingDirection();
		rate.sleep();
		ros::spinOnce();
	}

	return 0;
}
