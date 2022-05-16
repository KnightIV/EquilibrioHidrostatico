set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add source to this project's executable.
add_library(EQHS_Integrador 
					"Integrador.h" "Integrador.cpp" 
					"DataContainer.h" "DataContainer.cpp")