#pragma once

namespace eqhs_phys {

	constexpr double p_z0 = 680;			// pressure photosphere base; kg *m ^ -1 * s ^ -2
	constexpr double k_B = 1.380649e-23;	// Boltzmann const.; m ^ 2 * kg * s ^ -2 * K ^ -1
	constexpr double t_cor = 1.2e6;			// coronal temperature; K
	constexpr double t_phot = 6'000;		// photosphere temperature; K
	constexpr double z_t = 2e6;				// photosphere - corona transition altitude; m
	constexpr double z_w = 0.2e6;			// transition region width; m
	constexpr double sun_g = 274;			// Sun surface gravity acceleration; m *s ^ -2
	constexpr double m_p = 1.672621e-27;	// proton mass; kg

	constexpr double dtc = t_phot / t_cor;
}