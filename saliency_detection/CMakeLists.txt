cmake_minimum_required(VERSION 2.8.3)
project(saliency_detection)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg image_transport cv_bridge)

## Declare a catkin package
catkin_package()

## Build 
include_directories(include ${catkin_INCLUDE_DIRS})

# Add OpenCV 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Create executables
add_executable(saliencyDetectionHou src/saliencyDetectionHou.cpp)
target_link_libraries(saliencyDetectionHou ${OpenCV_LIBS} ${catkin_LIBRARIES})
add_executable(saliencyDetectionRudinac src/saliencyDetectionRudinac.cpp)
target_link_libraries(saliencyDetectionRudinac ${OpenCV_LIBS} ${catkin_LIBRARIES})
add_executable(saliencyDetectionItti src/saliencyDetectionItti.cpp src/cvgabor.cpp)
target_link_libraries(saliencyDetectionItti ${OpenCV_LIBS} ${catkin_LIBRARIES})
