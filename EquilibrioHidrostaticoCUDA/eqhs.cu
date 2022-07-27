#include "cuda_common.cuh"

#include <iostream>

#include "eqhs_phys.cuh"

using std::cout;
using std::endl;

struct SimProps {

	const double gridStep = 1'000;
	const double z_start = 0;
	const double z_end = 6e6;

	inline long gridSize() {
		return ((long)(z_end - z_start)) / (long)gridStep;
	}
};

__global__ void initAltitudeGrid(const SimProps *p, double *z) {
	int gid = (blockIdx.x * blockDim.x) + threadIdx.x;
	z[gid] = p->gridStep * gid;
}

__global__ void integrate(const SimProps *p, const double *z, double *temperature, double *pressure, double *density) {
	
}

int main() {
	SimProps p;
	const long sizeBytes = p.gridSize() * sizeof(double);

	SimProps *d_p;
	double *d_z;

	double *d_temperature, *d_pressure, *d_density;
	gpuErrCheck(cudaMalloc((void **)&d_temperature, sizeBytes));
	gpuErrCheck(cudaMalloc((void **)&d_pressure, sizeBytes));
	gpuErrCheck(cudaMalloc((void **)&d_density, sizeBytes));

	gpuErrCheck(cudaMalloc((void**) &d_p, sizeof(SimProps)));
	gpuErrCheck(cudaMemcpy(d_p, &p, sizeof(SimProps), cudaMemcpyHostToDevice));
	gpuErrCheck(cudaMalloc((void**) &d_z, sizeBytes));
	
	dim3 block(WARP_SIZE * 16);
	dim3 grid((p.gridSize() / block.x) + 1);
	//dim3 block(1), grid(1);
	initAltitudeGrid << <grid, block >> > (d_p, d_z);
	gpuErrCheck(cudaDeviceSynchronize());

	integrate << <grid, block >> > (d_p, d_z, d_temperature, d_pressure, d_density);
	gpuErrCheck(cudaDeviceSynchronize());

	gpuErrCheck(cudaFree(d_p));
	gpuErrCheck(cudaFree(d_z));
	gpuErrCheck(cudaFree(d_temperature));
	gpuErrCheck(cudaFree(d_pressure));
	gpuErrCheck(cudaFree(d_density));
	return 0;
}