cmake_minimum_required (VERSION 3.2)
project (CPUCONV)

set (CMAKE_CXX_STANDARD 11)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake/module)

file (GLOB SOURCES src/*.cpp)
file (GLOB HEADERS include/*.hpp)

add_executable (cpu_conv ${SOURCES} ${HEADERS})
target_link_libraries (cpu_conv)
