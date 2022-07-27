#pragma once

#include "cuda_common.cuh"
#include "math.h"

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

	__device__ double temperature(const double z) {
		double fac1 = (t_cor / 2);
		double fac2_tanh_inner = (z - z_t) / z_w;
		double tanh_exprsn = (1 - dtc) * tanh(fac2_tanh_inner);

		return fac1 * (1 + dtc + tanh_exprsn);
	}

	__device__ double temperatureIntegral(const double z) {
		double num_fac1 = (dtc - 1) * z_w;
		double exp_inner = ((2 * z_t) / z_w) - ((2 * z) / z_w);
		double ln_inner = dtc * exp(exp_inner) + 1;
		double den = 2 * t_cor * dtc;

		return ((num_fac1 * log(ln_inner)) / den) + (z / t_cor);
	}

	__device__ double pressure(const double temp_z) {
		double exp_innerfac2_frac = (m_p * sun_g) / (2 * k_B);
		double exp_fac = exp(temperatureIntegral(temp_z) * (-exp_innerfac2_frac));

		return p_z0 * exp_fac;
	}

	__device__ double density(const double temp_z, const double pressure_z) {
		double num = m_p * pressure_z;
		double den = 2 * k_B * temp_z;

		return num / den;
	}
}