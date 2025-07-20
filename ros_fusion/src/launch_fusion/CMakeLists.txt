cmake_minimum_required(VERSION 3.8)
project(launch_fusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch_fusion
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
