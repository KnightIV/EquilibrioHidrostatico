#pragma once

#include <tuple>
#include <memory>
#include <vector>
#include <string>
#include <math.h>

namespace eqhs_integrador {

	class AltitudeFunction {
	public:
		//const double* values;
		//const double* altitudes;
		const std::shared_ptr<std::vector<double>> altitudes;
		const std::shared_ptr<std::vector<double>> values;

	public:
		//AltitudeFunction(const double* altitudes, const double* values, const size_t numVals);
		AltitudeFunction(std::shared_ptr<std::vector<double>> altitudes, std::shared_ptr<std::vector<double>> values);

		size_t size();
		std::tuple<int, double> find_val_at_z(double z);

		std::tuple<double, double> operator[](int index);
	};
}