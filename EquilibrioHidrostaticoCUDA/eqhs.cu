#include "cuda_common.cuh"

__global__ void kern() {
	printf("Hello CUDA\n");
}

int main() {
	kern << <dim3(1), dim3(1) >> > ();
	gpuErrCheck(cudaDeviceSynchronize());
	return 0;
}