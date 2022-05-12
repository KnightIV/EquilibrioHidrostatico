# CMakeList.txt : CMake project for EquilibrioHidrostatico, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(EQHS_Integrador)

# Add source to this project's executable.
add_library(EQHS_Integrador 
					"Integrador.h" "Integrador.cpp")
