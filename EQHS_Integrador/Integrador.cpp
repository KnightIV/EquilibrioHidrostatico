#include "Integrador.h"

using namespace std;
using namespace eqhs_integrador;

double _scale_height(double temperature_z) {
	return (physical_constants::k_B * temperature_z) / (physical_constants::m_p * physical_constants::g);
}

double _mass_density(double p_z, double scale_height_z) {
	return p_z / (physical_constants::g * scale_height_z);
}

double _trapezoid_integral(double z1, double fn_z1, double z2, double fn_z2) {
	double z_height = abs(z2 - z1);
	return (z_height / 2) * (fn_z1 + fn_z2);
}

/// <summary>
/// Performs boileplate operations for getting the scale height at altitude z. Pushes scale height to vector of values.
/// </summary>
/// <param name="temperatureFn"></param>
/// <param name="index"></param>
/// <param name="scaleHeightVals"></param>
/// <returns>Tuple: <0> altitude (z); <1> scale height at z</returns>
tuple<double, double> _get_scale_height_index(AltitudeFunction& temperatureFn, int index, shared_ptr<vector<double>> scaleHeightVals) {
	tuple<double, double> temperatureTuple = temperatureFn[index];
	double z = get<0>(temperatureTuple);
	double temperature_z = get<1>(temperatureTuple);
	double scale_height_z = _scale_height(temperature_z);
	scaleHeightVals->push_back(scale_height_z);

	return { z, scale_height_z };
}

tuple<AltitudeFunction, AltitudeFunction, AltitudeFunction> eqhs_integrador::equilibrium_equations_values(AltitudeFunction& temperatureFn) {
	const double tolerance = 1e-5;

	auto scale_height_vals = make_shared<vector<double>>();
	auto pressure_vals = make_shared<vector<double>>();
	auto density_vals = make_shared<vector<double>>();
	auto altitude = temperatureFn.altitudes;

	int z0_index;
	double temperature_z0;
	tie(z0_index, temperature_z0) = temperatureFn.find_val_at_z(physical_constants::z_0);
	cout << "Reference temperature: (" << get<0>(temperatureFn[z0_index]) << " m, " << temperature_z0 << " K)\n";

	// add values at z0
	double sh_z0;
	tie(std::ignore, sh_z0) = _get_scale_height_index(temperatureFn, z0_index, scale_height_vals);
	pressure_vals->push_back(physical_constants::p_z0);
	density_vals->push_back(_mass_density(physical_constants::p_z0, sh_z0));

	// left of z_reference integrals
	double left_integral = 0;
	for (int i = z0_index - 1; i >= 0; i--) {
		double z_prev = get<0>(temperatureFn[i + 1]);
		double scale_height_z_prev = scale_height_vals->at(scale_height_vals->size() - 1);

		double z_cur, scale_height_z_cur;
		tie(z_cur, scale_height_z_cur) = _get_scale_height_index(temperatureFn, i, scale_height_vals);

		left_integral += _trapezoid_integral(z_prev, 1.0 / scale_height_z_prev, z_cur, 1.0 / scale_height_z_cur);
		double p_z = physical_constants::p_z0 * exp((-1) * left_integral);
		pressure_vals->push_back(p_z);

		density_vals->push_back(_mass_density(p_z, scale_height_z_cur));
	}

	reverse(scale_height_vals->begin(), scale_height_vals->end());
	reverse(pressure_vals->begin(), pressure_vals->end());
	reverse(density_vals->begin(), density_vals->end());

	// right of z_reference integrals
	double right_integral = 0;
	for (int i = z0_index + 1; i < temperatureFn.size(); i++) {
		double z_prev = get<0>(temperatureFn[i - 1]);
		double scale_height_z_prev = scale_height_vals->at(scale_height_vals->size() - 1);

		double z_cur, scale_height_z_cur;
		tie(z_cur, scale_height_z_cur) = _get_scale_height_index(temperatureFn, i, scale_height_vals);

		right_integral += _trapezoid_integral(z_prev, 1.0 / scale_height_z_prev, z_cur, 1.0 / scale_height_z_cur);
		double p_z = physical_constants::p_z0 * exp((-1) * right_integral);
		pressure_vals->push_back(p_z);

		density_vals->push_back(_mass_density(p_z, scale_height_z_cur));
	}

	return { AltitudeFunction(altitude, scale_height_vals),
				AltitudeFunction(altitude, pressure_vals),
				AltitudeFunction(altitude, density_vals) };
}
