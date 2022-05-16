#pragma once

#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <vector>
#include <sstream>
#include <memory>

namespace eqhs_data {

	const std::string TEMPERATURE_ALTITUDE_FILE_NAME = "Temperature-C7.dat";

	std::string get_data_directory();
	//std::tuple<double*, int, double*, int> get_alt_temp_values();
	std::tuple<std::shared_ptr<std::vector<double, std::allocator<double>>>, std::shared_ptr<std::vector<double, std::allocator<double>>>> get_alt_temp_values();
}