cmake_minimum_required(VERSION 2.8.3)
project(segmentation_3d)

find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  std_msgs
  visualization_msgs
  geometry_msgs
  image_transport
  sensor_msgs
  cv_bridge
)
find_package(PCL 1.7 REQUIRED)

catkin_package(
)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(
  ${catkin_INCLUDE_DIRS}
)

find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})
add_definitions(${EIGEN_DEFINITIONS})

add_executable(Extract_clusters src/segment_scene.cpp)
target_link_libraries(Extract_clusters ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(Extract_clusters gaze_msgs_generate_messages_cpp ${catkin_EXPORTED_TARGETS})
