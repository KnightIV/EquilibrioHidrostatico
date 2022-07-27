#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

constexpr auto WARP_SIZE = 32;

inline void gpuAssert(const cudaError_t errCode, const char* file, int line, bool abort = true) {
	if (errCode != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s \n File %s line %d",
			cudaGetErrorString(errCode), file, line);
		if (abort) {
			exit(errCode);
		}
	}
}

inline void gpuErrCheck(cudaError_t ans, bool abort = true) {
	gpuAssert(ans, __FILE__, __LINE__, abort);
}

__device__ inline const int calc1Dgid() {
	return (blockIdx.x * blockDim.x) + threadIdx.x;
}