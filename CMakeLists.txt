cmake_minimum_required(VERSION 3.10)
project(VisualOdometryExample)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV 4.0 REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src)

# Add source files
add_executable(monocular_vo
    src/main.cpp
    src/feature_tracker.cpp
    src/utils.cpp
)

# Link libraries
target_link_libraries(monocular_vo ${OpenCV_LIBS})
