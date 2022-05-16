#include "Integrador.h"

using namespace std;
using namespace eqhs_integrador;

double _scale_height(double temperature_z) {
	return (physical_constants::k_B * temperature_z) / (physical_constants::m_p * physical_constants::g);
}

AltitudeFunction eqhs_integrador::scale_height(AltitudeFunction& temperatureFn) {
	auto scale_height_vals = make_shared<vector<double>>();

	for (int i = 0; i < temperatureFn.size(); i++) {
		auto tup = temperatureFn[i];
		double z = get<0>(tup);
		double temp = get<1>(tup);

		scale_height_vals->push_back(_scale_height(temp));
	}

	return AltitudeFunction(temperatureFn.altitudes, scale_height_vals);
}
