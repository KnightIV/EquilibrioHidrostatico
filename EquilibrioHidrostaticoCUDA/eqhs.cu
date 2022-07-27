#include "cuda_common.cuh"

#include <iostream>

#include "eqhs_phys.cuh"

using std::cout;
using std::endl;

struct SimProps {

	const double gridStep = 1'000;
	const double z_start = 0;
	const double z_end = 6e6;

	SimProps() {}
	SimProps(const double gridStep, const double z_start, const double z_end)
		: gridStep(gridStep), z_start(z_start), z_end(z_end) {
	}

	inline long gridSize() {
		return ((long)(z_end - z_start)) / (long)gridStep;
	}
};

__global__ void initAltitudeGrid(const SimProps *p, double *z) {
	int gid = (blockIdx.x * blockDim.x) + threadIdx.x;
	z[gid] = p->gridStep * gid;
}

__global__ void calcTempIntegral(const SimProps *p, const double *z, double *temperature, double *temperatureIntegral) {
	
}

int main() {
	SimProps props;
	const long sizeBytes = props.gridSize() * sizeof(double);

	SimProps *d_props;
	double *d_z;

	double *d_temperature, *d_temperatureIntegral, *d_pressure, *d_density;
	gpuErrCheck(cudaMalloc((void **)&d_temperature, sizeBytes));
	gpuErrCheck(cudaMalloc((void **)&d_temperatureIntegral, sizeBytes));
	gpuErrCheck(cudaMalloc((void **)&d_pressure, sizeBytes));
	gpuErrCheck(cudaMalloc((void **)&d_density, sizeBytes));

	gpuErrCheck(cudaMalloc((void**) &d_props, sizeof(SimProps)));
	gpuErrCheck(cudaMemcpy(d_props, &props, sizeof(SimProps), cudaMemcpyHostToDevice));
	gpuErrCheck(cudaMalloc((void**) &d_z, sizeBytes));
	
	dim3 block(WARP_SIZE * 16);
	dim3 grid((props.gridSize() / block.x) + 1);
	//dim3 block(1), grid(1);
	initAltitudeGrid << <grid, block >> > (d_props, d_z);
	gpuErrCheck(cudaDeviceSynchronize());

	calcTempIntegral << <grid, block >> > (d_props, d_z, d_temperature, d_temperatureIntegral);
	gpuErrCheck(cudaDeviceSynchronize());

	gpuErrCheck(cudaFree(d_props));
	gpuErrCheck(cudaFree(d_z));
	gpuErrCheck(cudaFree(d_temperature));
	gpuErrCheck(cudaFree(d_temperatureIntegral));
	gpuErrCheck(cudaFree(d_pressure));
	gpuErrCheck(cudaFree(d_density));
	return 0;
}