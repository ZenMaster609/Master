cmake_minimum_required(VERSION 3.8)
project(gazebo_cone_plugins)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(eufs_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ignition-gazebo6 REQUIRED)
find_package(ignition-math6 REQUIRED)
find_package(ignition-plugin1 REQUIRED)

add_library(gazebo_ground_truth_cones SHARED
  src/gazebo_ground_truth_cones.cpp
)

ament_target_dependencies(gazebo_ground_truth_cones
  rclcpp
  eufs_msgs
)

target_link_libraries(gazebo_ground_truth_cones
  yaml-cpp
  ignition-gazebo6::core
  ignition-math6::ignition-math6
  ignition-plugin1::ignition-plugin1
)

install(TARGETS gazebo_ground_truth_cones
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.yaml"
)

ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/env_hooks/${PROJECT_NAME}.sh.in"
)

ament_package()
