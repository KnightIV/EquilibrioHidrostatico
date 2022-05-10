# CMakeList.txt : CMake project for EquilibrioHidrostatico, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

file(COPY "data" DESTINATION ".")

# Add source to this project's executable.
add_executable (EquilibrioHidrostatico 
						"EquilibrioHidrostatico.cpp" "EquilibrioHidrostatico.h" 
						"DataUtils.cpp" "DataUtils.h")
if(WIN32)
	set_property (TARGET EquilibrioHidrostatico
		PROPERTY
		# Enable C++17 standard compliance
		CXX_STANDARD 17)
endif()
