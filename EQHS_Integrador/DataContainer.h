#pragma once

#include <tuple>
#include <memory>
#include <vector>
#include <string>
#include <math.h>

namespace eqhs_integrador {

	class AltitudeFunction {
	public:
		const std::shared_ptr<std::vector<double>> altitudes;
		const std::shared_ptr<std::vector<double>> values;

	public:
		AltitudeFunction(std::shared_ptr<std::vector<double>> altitudes, std::shared_ptr<std::vector<double>> values);

		size_t size();

		/// <summary>
		/// Binary search through altitudes to find requested altitude (z). Assumes altitude vector is ordered.
		/// </summary>
		/// <param name="z">Altitude requested</param>
		/// <returns>[0] = index of altitude/value pair; [1] value of function at altitude</returns>
		std::tuple<int, double> find_val_at_z(double z);

		std::tuple<double, double> operator[](int index);
	};
}