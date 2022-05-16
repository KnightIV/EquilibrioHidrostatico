# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.13)

project ("EquilibrioHidrostatico")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include sub-projects.
add_subdirectory("EQHS_Integrador")
add_subdirectory("EQHS_Data")

add_executable(EquilibrioHidrostatico 
					"EquilibrioHidrostatico.cpp" "EquilibrioHidrostatico.h")

target_link_libraries(EquilibrioHidrostatico PUBLIC EQHS_Data)
target_link_libraries(EquilibrioHidrostatico PUBLIC EQHS_Integrador)

target_include_directories(EquilibrioHidrostatico PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/EQHS_Data"
                          "${PROJECT_SOURCE_DIR}/EQHS_Integrador")