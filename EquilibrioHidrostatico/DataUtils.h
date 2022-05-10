#pragma once

#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <map>
#include <tuple>
#include <vector>
#include <sstream>

namespace eqhs {

	std::string get_data_directory();
	std::tuple<double*, int, double*, int> get_alt_temp_values();
}