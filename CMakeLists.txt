cmake_minimum_required(VERSION 3.10)

project(demo)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(RKNN REQUIRED)
message(STATUS "RKNN INCLUDE DIR: "${RKNN_INCLUDE_DIRS})
include_directories(${RKNN_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE FRAMEWORK_SRC ${PROJECT_SOURCE_DIR}/src/*)

add_executable(demo main.cpp ${FRAMEWORK_SRC})
target_link_libraries(demo ${OpenCV_LIBS} ${RKNN_LIBS})
