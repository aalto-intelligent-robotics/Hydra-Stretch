cmake_minimum_required(VERSION 3.16)
project(hydra_stretch)

find_package(hydra REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  hydra_ros
  roscpp
  rospy
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS scripts/play_stretch_oneformer.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
