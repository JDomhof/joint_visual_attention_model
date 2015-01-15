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
#include "gaze_msgs/VectorPointcloudsClusters.h"
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
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <geometry_msgs/Point.h>
//
#include "pcl_ros/point_cloud.h"
//
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
class ExtractObjectClusters {
protected:
	boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_;
public:
	ros::Publisher pub,pub_cloud,pub_cloudVector, pub_centroid_objects;
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
	Mat imageKinect;

	geometry_msgs::PointStamped gaze_direction_out_msg;
	sensor_msgs::CameraInfo cameraInfo;
	pcl::PointCloud<pcl::PointXYZ> cloud;
	pcl::PointCloud<pcl::PointXYZ> cloudCone;

	bool b2Dviewer;
	bool b3Dviewer;
	double dPassTroughDistance;

	pcl::PointCloud<pcl::PointXYZ> prev_cloud_;
	pcl::PointCloud<pcl::Normal> prev_normals_;
	std::vector<pcl::PlanarRegion<pcl::PointXYZ>, Eigen::aligned_allocator<pcl::PlanarRegion<pcl::PointXYZ> > > prev_regions_;
	pcl::PointCloud<pcl::PointXYZ>::CloudVectorType prev_clusters_;
	size_t previous_data_size_;
	size_t previous_clusters_size_;
	visualization_msgs::Marker skelet;

	ExtractObjectClusters()
	{
		ros::NodeHandle private_nh("~");
		private_nh.param("TwoDviewer", b2Dviewer, false);
		private_nh.param("ThreeDviewer", b3Dviewer, false);
		private_nh.param("passTroughDistance", dPassTroughDistance, 2.0);

		previous_data_size_ = 0;
		previous_clusters_size_ = 0;

		m.resize(3,4);

		pub_cloud= nh.advertise <pcl::PointCloud<pcl::PointXYZ> > ("/segmenting/cloud", 1);
		pub_cloudVector= nh.advertise <gaze_msgs::VectorPointcloudsClusters> ("/segmenting/clusters", 1);
		pub_centroid_objects =nh.advertise<geometry_msgs::Point>("/segmenting/centroid_objects", 1);

		image_transport::ImageTransport it(nh);
		pubimage = it.advertise("/segmenting/mask", 1);

		sub_camera_info = nh.subscribe("/camera/rgb/camera_info", 1, &ExtractObjectClusters::callbackCameraInfo, this);
		sub_image_kinect = nh.subscribe("/camera/rgb/image_raw", 1, &ExtractObjectClusters::callbackKinect, this);
		sub_pcl = nh.subscribe("/camera/depth_registered/points", 1, &ExtractObjectClusters::callbackPCL, this);

		if(b3Dviewer) {
			vis_ = createVis();
		}
	}
	void callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& info);
	void callbackKinect(const sensor_msgs::ImageConstPtr& image1);
	void callbackPCL(const sensor_msgs::PointCloud2ConstPtr& cloud_in);

	void mainLoop();
	void pointingDetect();
	void pointingDetectCone();
	void pointingCreatePoingintMap();
	void findBoundingBoxGaze(Mat img);
	void findBoundingBoxPointing(Mat img, Point p);
	void findEnclosedGoxGaze(Mat img);
	void findEnclosedBoxPointing(Mat img, Point p);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> createVis();
};
boost::shared_ptr<pcl::visualization::PCLVisualizer> ExtractObjectClusters::createVis() {
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

	viewer->setBackgroundColor (0, 255, 255);
	//viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();
	viewer->setCameraPosition (0, 0, -1.5, 0, -0.5, 3, 0, -1, -0);
	return viewer;
}
void ExtractObjectClusters::callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& info) {
	cameraInfo = *info;
	m << cameraInfo.P[0],cameraInfo.P[1],cameraInfo.P[2],cameraInfo.P[3],cameraInfo.P[4],cameraInfo.P[5],cameraInfo.P[6],cameraInfo.P[7],cameraInfo.P[8],cameraInfo.P[9],cameraInfo.P[10],cameraInfo.P[11];
}
void ExtractObjectClusters::callbackPCL(const sensor_msgs::PointCloud2ConstPtr& cloud_in) {
	pcl::PCLPointCloud2 pcl_pc;
	pcl_conversions::toPCL(*cloud_in, pcl_pc);
	pcl::fromPCLPointCloud2(pcl_pc, cloud);
}
void ExtractObjectClusters::callbackKinect(const sensor_msgs::ImageConstPtr& image1) {
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
void ExtractObjectClusters::mainLoop() {
	clock_t start = clock();
	//If pointcloud is available
	if(cloud.points.size() > 0) {
		//---------------------Remove background -------------------------------
		pcl::PointCloud<pcl::PointXYZ> cloud_filtered;
		cloud_filtered.height = cloud.height;
		cloud_filtered.width = cloud.width;

		pcl::PassThrough<pcl::PointXYZ> pass;
		pass.setInputCloud (cloud.makeShared());
		pass.setFilterFieldName ("z");
		pass.setFilterLimits (0.0, dPassTroughDistance);
		pass.setKeepOrganized(true);
		//pass.setFilterLimitsNegative (true);
		pass.filter (cloud_filtered);

		//-----------------Normal Estimation---------------------------
		pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
		ne.setMaxDepthChangeFactor (0.02f);
		ne.setNormalSmoothingSize (20.0f);
		pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
		ne.setInputCloud (cloud_filtered.makeShared());
		ne.compute (*normal_cloud);

		//-----------------Planar Detection ---------------------------
		// Segment Planes
		std::vector<pcl::PlanarRegion<pcl::PointXYZ>, Eigen::aligned_allocator<pcl::PlanarRegion<pcl::PointXYZ> > > regions;
		std::vector<pcl::ModelCoefficients> model_coefficients;
		std::vector<pcl::PointIndices> inlier_indices;
		pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
		std::vector<pcl::PointIndices> label_indices;
		std::vector<pcl::PointIndices> boundary_indices;
		pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZ, pcl::Normal, pcl::Label> mps;
		mps.setInputNormals (normal_cloud);
		mps.setInputCloud (cloud_filtered.makeShared());
		mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);

		//---------------- Planar Removal------------------------------
		//Segment Objects
		pcl::PointCloud<pcl::PointXYZ>::CloudVectorType clusters;
		pcl::EuclideanClusterComparator<pcl::PointXYZ, pcl::Normal, pcl::Label>::Ptr euclidean_cluster_comparator_;
		euclidean_cluster_comparator_ = pcl::EuclideanClusterComparator<pcl::PointXYZ, pcl::Normal, pcl::Label>::Ptr (new pcl::EuclideanClusterComparator<pcl::PointXYZ, pcl::Normal, pcl::Label> ());

		//Only continue if there is a planar surface, because the assumption is made that there is at least one planar surface
		Mat maskWithAllObjects = Mat::zeros(imageKinect.rows,imageKinect.cols, CV_8UC1);
		if (regions.size () > 0) {
			std::vector<bool> plane_labels;
			plane_labels.resize (label_indices.size (), false);
			for (size_t i = 0; i < label_indices.size (); i++)
			{
				if (label_indices[i].indices.size () > 2000)  //TODO: Determine correct number of required indices
				{
					plane_labels[i] = true;
				}
			}

			euclidean_cluster_comparator_->setInputCloud (cloud_filtered.makeShared());
			euclidean_cluster_comparator_->setLabels (labels);
			euclidean_cluster_comparator_->setExcludeLabels (plane_labels);
			euclidean_cluster_comparator_->setDistanceThreshold (0.1f, false); //TODO: check threshold

			pcl::PointCloud<pcl::Label> euclidean_labels;
			std::vector<pcl::PointIndices> euclidean_label_indices;
			pcl::OrganizedConnectedComponentSegmentation<pcl::PointXYZ,pcl::Label> euclidean_segmentation (euclidean_cluster_comparator_);
			euclidean_segmentation.setInputCloud (cloud_filtered.makeShared());

			euclidean_segmentation.segment (euclidean_labels, euclidean_label_indices);

			gaze_msgs::VectorPointcloudsClusters vectorClustersObjects;
			pcl::PointCloud<pcl::PointXYZ> cloudOut;
			cloudOut.height = cloud.height;
			cloudOut.width = cloud.width;
			for (size_t i = 0; i < euclidean_label_indices.size (); i++) {
				if (euclidean_label_indices[i].indices.size () > 50) {
					//In clusters
					pcl::PointCloud<pcl::PointXYZ> cluster;
					pcl::copyPointCloud (cloud_filtered,euclidean_label_indices[i].indices,cluster);
					cloudOut += cluster;
					clusters.push_back(cluster);

					//Convert pointcloud to msg and add in vectorClustersObjects
					sensor_msgs::PointCloud2 clusterObject;
					pcl::toROSMsg(cluster, clusterObject);
					vectorClustersObjects.pointClouds.push_back(clusterObject);
				}
			}

			//-----------------Create 2D mask -----------------------------
			//Only continue if clusters are found
			if( cloudOut.size() > 0 ) {
				for (int iNumberOfIndices = 0; iNumberOfIndices < cloudOut.points.size(); iNumberOfIndices++) {
					//Transform 3D point to 2D
					Eigen::VectorXf coordinates3d(4);
					coordinates3d << cloudOut.points[iNumberOfIndices].x, cloudOut.points[iNumberOfIndices].y, cloudOut.points[iNumberOfIndices].z, 1.0;
					Eigen::Vector3f coordinates2d = m*coordinates3d;  coordinates2d = coordinates2d/coordinates2d.z();

					//Mask containing all objects
					if (!isnan(coordinates2d.x()) && !isnan(coordinates2d.y())) {
						maskWithAllObjects.at<uchar>(Point(coordinates2d.x(), coordinates2d.y())) = 255;
					}
				}
			}

			//Close mask to remove gaps
			int morph_size = 3;
			Mat element = getStructuringElement( 0, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
			morphologyEx(maskWithAllObjects, maskWithAllObjects, MORPH_CLOSE, element, Point(-1,-1), 5);

			//Plot results
			if(b2Dviewer) {
				imshow("maskWithAllObjects", maskWithAllObjects);
				Mat output;
				imageKinect.copyTo(output, maskWithAllObjects);
				imshow("objects found", output);

				//imshow("Clean", imageKinect);
				waitKey(3);
			}

			if (b3Dviewer){
				prev_cloud_ = cloud;
				prev_regions_ = regions;
				prev_clusters_ = clusters;

				removePreviousDataFromScreen (previous_data_size_, previous_clusters_size_, vis_);
				if (!vis_->updatePointCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(prev_cloud_), "cloud"))
				{
					vis_->addPointCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(prev_cloud_), "cloud");
				}

				displayPlanarRegions (prev_regions_, vis_);
				displayEuclideanClusters (prev_clusters_,vis_);

				previous_data_size_ = prev_regions_.size ();
				previous_clusters_size_ = prev_clusters_.size ();
				vis_->spinOnce();
			}

			//-------------Publish-------------------
			sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", maskWithAllObjects).toImageMsg();
			pubimage.publish(msg);

			pub_cloud.publish(cloudOut);

			Eigen::Vector4f centroid;
			pcl::compute3DCentroid(cloudOut, centroid);
			geometry_msgs::Point centroidObjects; centroidObjects.x = centroid.x(); centroidObjects.y = centroid.y(); centroidObjects.z = centroid.z();
			pub_centroid_objects.publish(centroidObjects);

			pub_cloudVector.publish(vectorClustersObjects);
		}

		clock_t end = clock();
		cout << "Segmenting took: " << (float)(end - start) / CLOCKS_PER_SEC << " seconds" << endl;
	}
}
int main(int argc, char** argv)
{
	ros::init(argc, argv, "Extract_object_clusters");

	ExtractObjectClusters object;
	ros::Rate rate(5); // 5Hz

	while (ros::ok())
	{
		object.mainLoop();
		rate.sleep();
		ros::spinOnce();
	}

	return 0;
}
