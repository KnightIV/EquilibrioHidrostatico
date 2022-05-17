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

	class FinalExportData {
	public:
		const std::shared_ptr<std::vector<double>> altitudes;
		const std::shared_ptr<std::vector<double>> temperature;
		const std::shared_ptr<std::vector<double>> pressure;
		const std::shared_ptr<std::vector<double>> density;
		const std::shared_ptr<std::vector<double>> scaleHeight;

	public:
		FinalExportData(std::shared_ptr<std::vector<double>> altitudes,
			std::shared_ptr<std::vector<double>> temperature,
			std::shared_ptr<std::vector<double>> pressure,
			std::shared_ptr<std::vector<double>> density,
			std::shared_ptr<std::vector<double>> scaleHeight);

		size_t size() const;
	};

	std::string get_data_directory();
	std::tuple<std::shared_ptr<std::vector<double>>, std::shared_ptr<std::vector<double>>> get_alt_temp_values();
	void export_data_csv(const FinalExportData& exportData, const std::string outputFileName);
}