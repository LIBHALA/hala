#!/usr/bin/env bash

sPWD=`pwd`
if [ $sPWD != @CMAKE_CURRENT_BINARY_DIR@ ]; then
    echo "NOPE: you must run this inside @CMAKE_CURRENT_BINARY_DIR@"
    exit 1;
fi

mkdir -p FinalTest
cd FinalTest || { echo "ERROR: Could not cd into FinalTest, aborting"; exit 1; }

sPWD=`pwd`
if [ $sPWD != "@CMAKE_CURRENT_BINARY_DIR@/FinalTest" ]; then
    echo "ERROR: somehow we failed to cd into FinalTest"
    exit 1;
fi

rm -fr ../FinalTest/*

cat > test.cpp <<- TEST_FILE
#include <iostream>
#include "hala.hpp"

int main(int, char**){

    std::vector<int> pntr = {0, 1, 2, 3};
    std::vector<int> indx = {0, 1, 2};
    std::vector<double> vals = {1.0, 2.0, 3.0};
    std::vector<double> r, x = {4.0, 5.0, 6.0};

    hala::sparse_gemv('N', 3, 3, 1.0, pntr, indx, vals, x, 0.0, r);

    if (r.size() != 3) throw std::runtime_error("hala failed to resize vector");
    if (r[0] !=  4.0) throw std::runtime_error("hala failed for entry 0");
    if (r[1] != 10.0) throw std::runtime_error("hala failed for entry 1");
    if (r[2] != 18.0) throw std::runtime_error("hala failed for entry 2");

    return 0;
}
TEST_FILE

cat > CMakeLists.txt <<- TEST_FILE_MAKE
cmake_minimum_required(VERSION 3.10)

project(TestHALA VERSION 0.1.0 LANGUAGES CXX)

find_package(HALA @HALA_VERSION_MAJOR@.@HALA_VERSION_MINOR@.@HALA_VERSION_PATCH@ PATHS "@CMAKE_INSTALL_PREFIX@" REQUIRED @HALA_COMPONENTS@)

add_executable(smoke_test test.cpp)
target_link_libraries(smoke_test HALA::HALA)
TEST_FILE_MAKE


mkdir -p Build
cd Build || { echo "ERROR: Could not cd into Build, aborting"; exit 1; }

sPWD=`pwd`
if [ $sPWD != "@CMAKE_CURRENT_BINARY_DIR@/FinalTest/Build" ]; then
    echo "ERROR: somehow we failed to cd into Build"
    exit 1;
fi

rm -fr ../Build/*

@CMAKE_COMMAND@ $1 ..
make
./smoke_test || { echo "ERROR: Could not run the post-install HALA test"; exit 1; }

if [[ @HALA_ENABLE_EXAMPLES@ == "ON" ]]; then
    set -e
    rm -fr ../Build/*
    @CMAKE_COMMAND@ $1 @CMAKE_INSTALL_PREFIX@/share/hala/examples/
    make
    ./example_gauss_seidel -fast
fi

echo ''
echo '######################################'
echo '    HALA Installation Check: GOOD'
echo '######################################'
