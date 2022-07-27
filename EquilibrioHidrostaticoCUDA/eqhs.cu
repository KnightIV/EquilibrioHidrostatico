#include "cuda_common.cuh"

#include <iostream>

using std::cout;
using std::endl;

struct SimProps {

	const double gridStep = 1'000;
	const double z_start = 0;
	const double z_end = 6e6;
};

__global__ void initAltitudeGrid(const SimProps *p, double *z) {
	printf("%f, [%f, %f]\n", p->gridStep, p->z_start, p->z_end);
}

int main() {
	SimProps p;
	SimProps *d_p;
	
	gpuErrCheck(cudaMalloc((void**) &d_p, sizeof(SimProps)));
	gpuErrCheck(cudaMemcpy(d_p, &p, sizeof(SimProps), cudaMemcpyHostToDevice));
	
	dim3 grid(1);
	dim3 block(1);
	initAltitudeGrid << <grid, block >> > (d_p, nullptr);

	gpuErrCheck(cudaDeviceSynchronize());
	
	gpuErrCheck(cudaFree(d_p));
	return 0;
}