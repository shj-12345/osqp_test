cmake_minimum_required(VERSION 3.10.0)
project(osqp-eigen_test)

find_package(OsqpEigen)
find_package(Eigen3)

include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

add_executable(MPCExample MPCExample.cpp) 
target_link_libraries(MPCExample OsqpEigen::OsqpEigen)

add_executable(simpleExample simpleExample.cpp) 
target_link_libraries(simpleExample OsqpEigen::OsqpEigen)

# 由于matplotlib是由python经过包装生成的c++库，因此需要链接python库
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)
include_directories(SYSTEM "/usr/include/python3.8")#包含python3的头文件路径
add_executable(quintic_spline_smoother quintic_spline_smoother.cpp) 
# target_link_libraries(quintic_spline_smoother PRIVATE ${Python3_LIBRARIES}  OsqpEigen::OsqpEigen)
target_link_libraries(quintic_spline_smoother PRIVATE 
Python3::Python 
Python3::Module 
Python3::NumPy 
OsqpEigen::OsqpEigen)

#find_package用法
# find_package(PCL 1.2 REQUIRED)
# MESSAGE(STATUS "PCL_FOUND=${PCL_FOUND}")
# MESSAGE(STATUS "PCL_INCLUDE_DIRS=${PCL_INCLUDE_DIRS}")
# MESSAGE(STATUS "PCL_LIBRARIES=${PCL_LIBRARIES}")
# MESSAGE(STATUS "PCL_COMPONENTS=${PCL_COMPONENTS}")