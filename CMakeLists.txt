cmake_minimum_required(VERSION 3.1.0)

project(motion_detector)

find_package(PCL     REQUIRED)
find_package(Eigen3  REQUIRED)
find_package(OpenMP  REQUIRED)
find_package(livox_msg REQUIRED)

fins_add_node(${PROJECT_NAME}
  node.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
  include
  ${PCL_INCLUDE_DIRS}
)

fins_link_ros_dependencies(${PROJECT_NAME}
  geometry_msgs
  nav_msgs
  sensor_msgs
  pcl_conversions
  tf2_eigen
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenMP::OpenMP_CXX
    ${PCL_LIBRARIES}
    pcl_surface
    livox_msg::livox_msg
)


message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")

# 改为 target...

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 5)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=4)
    message("core for MP:  3")
  elseif(N GREATER 3)
    math(EXPR PROC_NUM "${N} - 2")
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM="${PROC_NUM}")
    message("core for MP:  ${PROC_NUM}")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()
