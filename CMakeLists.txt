cmake_minimum_required(VERSION 3.0.2)
project(hydra_stretch)

find_package(catkin REQUIRED COMPONENTS
  hydra
  hydra_ros
  kimera_semantics
  roscpp
  rospy
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS scripts/play_stretch_oneformer.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
