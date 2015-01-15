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

#include <math.h>
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include <image_transport/image_transport.h>//Use image_transport for publishing and subscribing to images in ROS

#include <geometry_msgs/PointStamped.h>
#include "gaze_msgs/PointingDirection.h"
#include <Eigen/Dense>

#include <vector>
#include <numeric>

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
#include <pcl/impl/point_types.hpp>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/filter.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/features/organized_edge_detection.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

using namespace cv;
using namespace std;

#define PI 3.14159265
RNG rng(12345);

bool enforceIntensitySimilarity (const pcl::PointXYZI& point_a, const pcl::PointXYZI& point_b, float squared_distance)
{
  if (fabs (point_a.intensity - point_b.intensity) < 50.0f)
    return (true);
  else
    return (false);
}
void
displayPlanarRegions (std::vector<pcl::PlanarRegion<pcl::PointXYZ>, Eigen::aligned_allocator<pcl::PlanarRegion<pcl::PointXYZ> > > &regions,
                      boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
  char name[1024];
  unsigned char red [6] = {255,   0,   0, 255, 255,   0};
  unsigned char grn [6] = {  0, 255,   0, 255,   0, 255};
  unsigned char blu [6] = {  0,   0, 255,   0, 255, 255};

  pcl::PointCloud<pcl::PointXYZ>::Ptr contour (new pcl::PointCloud<pcl::PointXYZ>);

  for (size_t i = 0; i < regions.size (); i++)
  {

    Eigen::Vector3f centroid = regions[i].getCentroid ();
    Eigen::Vector4f model = regions[i].getCoefficients ();
    pcl::PointXYZ pt1 = pcl::PointXYZ (centroid[0], centroid[1], centroid[2]);
    pcl::PointXYZ pt2 = pcl::PointXYZ (centroid[0] + (0.5f * model[0]),
                                       centroid[1] + (0.5f * model[1]),
                                       centroid[2] + (0.5f * model[2]));
    sprintf (name, "normal_%d", unsigned (i));
    viewer->addArrow (pt2, pt1, 1.0, 0, 0, false, name);

    contour->points = regions[i].getContour ();
    sprintf (name, "plane_%02d", int (i));
    pcl::visualization::PointCloudColorHandlerCustom <pcl::PointXYZ> color (contour, red[i%6], grn[i%6], blu[i%6]);
    if(!viewer->updatePointCloud(contour, color, name))
      viewer->addPointCloud (contour, color, name);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, name);
  }
}
void
removePreviousDataFromScreen (size_t prev_models_size, size_t prev_clusters_size, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
  char name[1024];
  for (size_t i = 0; i < prev_models_size; i++)
  {
    sprintf (name, "normal_%d", unsigned (i));
    viewer->removeShape (name);

    sprintf (name, "plane_%02d", int (i));
    viewer->removePointCloud (name);
  }

  for (size_t i = 0; i < prev_clusters_size; i++)
  {
    sprintf (name, "cluster_%d", int (i));
    viewer->removePointCloud (name);
  }
}
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
class CreatePointingMap {
protected:
	boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_;
public:
	ros::Subscriber sub_image_kinect, sub_image_eyetracker, sub_gaze, sub_image_saliency,sub_pointing_dir, sub_pcl, sub_camera_info, sub_pcl_cone;
	ros::Subscriber sub_markers;
	ros::NodeHandle nh;
	image_transport::Publisher pubimage;

	Point positionFinger;
	Point vectorFinger;
	Eigen::Vector3f positionFinger3D;
	Eigen::Vector3f vectorFinger3D;
    Eigen::MatrixXf m;

	Point2f gaze_direction_eyetracker; //Coming from eyetracker, to be transformed to Kinect frame
	std::vector<Point2f> previous_gaze;

	Mat saliencyMap;
	Point2f centerObject;
	Mat imageKinect;

	geometry_msgs::PointStamped gaze_direction_out_msg;
	sensor_msgs::CameraInfo cameraInfo;
	pcl::PointCloud<pcl::PointXYZ> cloud;
	pcl::PointCloud<pcl::PointXYZ> cloudCone;

	double dStdProbabilityMap;
	double dPassTroughDistance;
	bool b2Dviewer;

	pcl::PointCloud<pcl::PointXYZ> prev_cloud_;
	pcl::PointCloud<pcl::Normal> prev_normals_;
	std::vector<pcl::PlanarRegion<pcl::PointXYZ>, Eigen::aligned_allocator<pcl::PlanarRegion<pcl::PointXYZ> > > prev_regions_;
	pcl::PointCloud<pcl::PointXYZ>::CloudVectorType prev_clusters_;
    size_t previous_data_size_;
    size_t previous_clusters_size_;
    visualization_msgs::Marker skelet;

    CreatePointingMap()
	{
		ros::NodeHandle private_nh("~");
		private_nh.param("passTroughDistance", dPassTroughDistance, 2.0);
		private_nh.param("TwoDviewer", b2Dviewer, false);
		private_nh.param("stdProbabilityMap", dStdProbabilityMap, 25.0);

		previous_data_size_ = 0;
		previous_clusters_size_ = 0;

		m.resize(3,4);

		image_transport::ImageTransport it(nh);
		pubimage = it.advertise("/pointing/probabilitymap", 1);

		sub_camera_info = nh.subscribe("/camera/rgb/camera_info", 1, &CreatePointingMap::callbackCameraInfo, this);
		sub_pointing_dir = nh.subscribe("/pointing/direction", 1, &CreatePointingMap::callbackPointing, this);
		sub_image_kinect = nh.subscribe("/camera/rgb/image_raw", 1, &CreatePointingMap::callbackKinect, this);
		sub_pcl = nh.subscribe("/segmenting/cloud", 1, &CreatePointingMap::callbackPCL, this);
		sub_markers = nh.subscribe("/skeleton_markers", 1, &CreatePointingMap::callbackSkelet, this);
	}
	void callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& info);
	void callbackPointing(const gaze_msgs::PointingDirectionConstPtr& pointingDir);
	void callbackKinect(const sensor_msgs::ImageConstPtr& image1);
	void callbackPCL(const sensor_msgs::PointCloud2ConstPtr& cloud_in);
	void callbackSkelet(const visualization_msgs::MarkerConstPtr& markers);

	void mainLoop();
	pcl::PointCloud<pcl::PointXYZ> removeBody();
	Mat computeProbabilityMap(pcl::PointCloud<pcl::PointXYZ> cloud_remove_body);
	void plotAndPublish(Mat probabilityMap);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> createVis();
};
boost::shared_ptr<pcl::visualization::PCLVisualizer> CreatePointingMap::createVis() {
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

	viewer->setBackgroundColor (0, 255, 255);
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();
	viewer->setCameraPosition (0, 0, -1.5, 0, -0.5, 3, 0, -1, -0);
	return viewer;
}
void CreatePointingMap::callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& info) {
	cameraInfo = *info;
	m << cameraInfo.P[0],cameraInfo.P[1],cameraInfo.P[2],cameraInfo.P[3],cameraInfo.P[4],cameraInfo.P[5],cameraInfo.P[6],cameraInfo.P[7],cameraInfo.P[8],cameraInfo.P[9],cameraInfo.P[10],cameraInfo.P[11];
}
void CreatePointingMap::callbackPointing(const gaze_msgs::PointingDirectionConstPtr& pointingDir) {
	gaze_msgs::PointingDirection in = (*pointingDir);
	positionFinger  = Point(in.position_2d.x, in.position_2d.y);
	vectorFinger  = Point(in.vector_2d.x, in.vector_2d.y);

	positionFinger3D << in.position_3d.x, in.position_3d.y, in.position_3d.z;
	vectorFinger3D << in.vector_3d.x, in.vector_3d.y, in.vector_3d.z;
}
void CreatePointingMap::callbackKinect(const sensor_msgs::ImageConstPtr& image1) {
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
void CreatePointingMap::callbackPCL(const sensor_msgs::PointCloud2ConstPtr& cloud_in) {
	pcl::PCLPointCloud2 pcl_pc;
	pcl_conversions::toPCL(*cloud_in, pcl_pc);
	pcl::fromPCLPointCloud2(pcl_pc, cloud);
}
void CreatePointingMap::callbackSkelet(const visualization_msgs::MarkerConstPtr& markers) {
	skelet = *markers;
}
pcl::PointCloud<pcl::PointXYZ> CreatePointingMap::removeBody() {
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
Mat CreatePointingMap::computeProbabilityMap(pcl::PointCloud<pcl::PointXYZ> cloud_remove_body) {
	Mat probabilityMap = Mat::zeros(imageKinect.rows,imageKinect.cols, CV_8UC1);

	//Only continue if size > 0
	if( cloud_remove_body.size() > 0 ) {
		for (int iNumberOfIndices = 0; iNumberOfIndices < cloud_remove_body.points.size(); iNumberOfIndices++) {
			//Transform 3D point to 2D
			Eigen::VectorXf coordinates3d(4);
			coordinates3d << cloud_remove_body.points[iNumberOfIndices].x, cloud_remove_body.points[iNumberOfIndices].y, cloud_remove_body.points[iNumberOfIndices].z, 1.0;
			Eigen::Vector3f coordinates2d = m*coordinates3d;  coordinates2d = coordinates2d/coordinates2d.z();

			//Determine vector from position finger to point pointcloud
			Eigen::Vector3f objectPoint;
			objectPoint << cloud_remove_body.points[iNumberOfIndices].x, cloud_remove_body.points[iNumberOfIndices].y, cloud_remove_body.points[iNumberOfIndices].z;
			objectPoint = objectPoint - positionFinger3D;

			if (objectPoint.norm() > 0.05) {
				//Compute angle with pointing vector
				double cosineAngle = objectPoint.dot(vectorFinger3D) / (objectPoint.norm()*(vectorFinger3D).norm());

				//Compute pointing map value
				double angleRadians = acos(cosineAngle);
				double sigma = dStdProbabilityMap*PI/180; //from degrees to radians
				if (angleRadians<2*sigma) { //because value is small < 0.1
					//Angle
					double value = exp (- (angleRadians*angleRadians/(2*sigma*sigma) ) );
					probabilityMap.at<uchar>(Point(coordinates2d.x(), coordinates2d.y())) = 255*value;
				}
			}
		}
	}
	return probabilityMap;
}
void CreatePointingMap::plotAndPublish(Mat probabilityMap) {
	//-------------------Close gaps and normalize-----------------------------:
	//Close mask to remove gaps
	int morph_size = 3;
	Mat element = getStructuringElement( 0, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
	morphologyEx(probabilityMap, probabilityMap, MORPH_CLOSE, element);

	//Normalize probability map
	bool bNormalize = true;
	if(bNormalize) {
		cv::normalize(probabilityMap, probabilityMap, 0, 255, NORM_MINMAX, CV_8UC1);
	}

	//Plot results
	if(b2Dviewer && (!probabilityMap.empty()) ) {
		imshow("Probability map", probabilityMap);
		waitKey(3);
	}

	//Publish results
	sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", probabilityMap).toImageMsg();
	pubimage.publish(msg);
}
void CreatePointingMap::mainLoop() {
	clock_t start = clock();
	if(skelet.points.size() > 0 && vectorFinger3D.size() >0) {
		//Remove body out of pointcloud to obtain all objects
		pcl::PointCloud<pcl::PointXYZ> cloud_filtered = removeBody();

		//Compute map
		Mat probabilityMap = computeProbabilityMap(cloud_filtered);

		//Plot and Publish
		plotAndPublish(probabilityMap);
	}
	clock_t end = clock();
	cout << "Creating Pointing Map frame took: " << (float)(end - start) / CLOCKS_PER_SEC << " seconds" << endl;
}
int main(int argc, char** argv)
{
  ros::init(argc, argv, "Create_pointing_map");

  CreatePointingMap object;
  ros::Rate rate(5); // 5Hz

  while (ros::ok())
  {
	  object.mainLoop();
	  rate.sleep();
	  ros::spinOnce();
  }

  return 0;
}
