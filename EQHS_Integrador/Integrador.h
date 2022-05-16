#pragma once

#include <iostream>
#include <tuple>
#include <math.h>
#include <algorithm>

#include "DataContainer.h"

namespace eqhs_integrador {

	namespace physical_constants {

		const double z_0 = 0;				// reference altitude (p_ref); m
		const double p_z0 = 680;			// pressure photosphere base; kg* m ^ -1 * s ^ -2 (Pa)
		const double k_B = 1.380649e-23;	// Boltzmann const.; m ^ 2 * kg * s ^ -2 * K ^ -1
		const double g = 274;				// Sun surface gravity acceleration; m* s ^ -2
		const double m_p = 1.672621e-27;	// proton mass; kg
	}

	/// <summary>
	/// Calculates equilibrium equations based on temperature field given
	/// </summary>
	/// <param name="temperatureFn">Temperature field as function of altitude</param>
	/// <returns>3 functions of altitude: [0] = scale height [1] = pressure [2] = mass density</returns>
	std::tuple<AltitudeFunction, AltitudeFunction, AltitudeFunction> equilibrium_equations_values(AltitudeFunction& scaleHeightFn);
}