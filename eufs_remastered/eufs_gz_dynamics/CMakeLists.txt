cmake_minimum_required(VERSION 3.8)
project(eufs_gz_dynamics)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(eufs_models REQUIRED)
find_package(ignition-gazebo6 REQUIRED)
find_package(ignition-math6 REQUIRED)
find_package(ignition-plugin1 REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/eufs_gz_dynamics.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  geometry_msgs
  eufs_models
)

target_link_libraries(${PROJECT_NAME}
  ignition-gazebo6::core
  ignition-math6::ignition-math6
  ignition-plugin1::ignition-plugin1
)

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib
)

ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/env_hooks/${PROJECT_NAME}.sh.in"
)

ament_package()
