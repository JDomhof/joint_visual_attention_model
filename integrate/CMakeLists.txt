cmake_minimum_required(VERSION 2.8.3)
project(integrate)

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
  gaze_msgs)

find_package(PCL 1.7 REQUIRED)

## Declare a catkin package
catkin_package()

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

## Build 
include_directories(${catkin_INCLUDE_DIRS})

add_executable(integrate src/integrate.cpp)
target_link_libraries(integrate ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(integrate gaze_msgs_generate_messages_cpp ${catkin_EXPORTED_TARGETS})
