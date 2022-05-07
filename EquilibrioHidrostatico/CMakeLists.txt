# CMakeList.txt : CMake project for EquilibrioHidrostatico, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

file(COPY "data" DESTINATION ".")

# Add source to this project's executable.
add_executable (EquilibrioHidrostatico 
						"EquilibrioHidrostatico.cpp" "EquilibrioHidrostatico.h" 
						"DataUtils.cpp" "DataUtils.h")

# TODO: Add tests and install targets if needed.
