#include "cuda_common.cuh"

#include <iostream>

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

int main() {
	SimProps p;

	SimProps *d_p;
	double *d_z;

	gpuErrCheck(cudaMalloc((void**) &d_p, sizeof(SimProps)));
	gpuErrCheck(cudaMemcpy(d_p, &p, sizeof(SimProps), cudaMemcpyHostToDevice));
	gpuErrCheck(cudaMalloc((void**) &d_z, p.gridSize() * sizeof(double)));
	
	dim3 block(WARP_SIZE * 16);
	dim3 grid((p.gridSize() / block.x) + 1);
	initAltitudeGrid << <grid, block >> > (d_p, d_z);
	gpuErrCheck(cudaDeviceSynchronize());
	
	gpuErrCheck(cudaFree(d_p));
	gpuErrCheck(cudaFree(d_z));
	return 0;
}