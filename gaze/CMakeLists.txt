cmake_minimum_required(VERSION 2.8.3)
project(gaze)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	rospy
	std_msgs
	genmsg 
	image_transport
	cv_bridge
    gaze_msgs
	geometry_msgs)

find_package(OpenCV REQUIRED)

## Declare a catkin package
catkin_package(
  INCLUDE_DIRS include
 )

## Build 
include_directories(
	include 
	${catkin_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
)

find_package( Eigen REQUIRED )
include_directories( ${EIGEN_INCLUDE_DIRS} )

add_executable(probabilityMapHD src/probabilityMapHD.cpp)
target_link_libraries(probabilityMapHD ${OpenCV_LIBS} ${catkin_LIBRARIES})
add_dependencies(probabilityMapHD ${catkin_EXPORTED_TARGETS})
