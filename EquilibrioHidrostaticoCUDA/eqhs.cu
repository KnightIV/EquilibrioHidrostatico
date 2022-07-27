#include "cuda_common.cuh"

#include <iostream>
#include <fstream>
#include <filesystem>

#include "eqhs_phys.cuh"

using std::cout;
using std::endl;
using std::string;
using std::ofstream;

struct SimProps {

	const double gridStep = 1'000;
	const double z_start = 0;
	const double z_end = 6e6;

	SimProps() {}
	SimProps(const double gridStep, const double z_start, const double z_end)
		: gridStep(gridStep), z_start(z_start), z_end(z_end) {
	}

	__host__ __device__ inline const long gridSize() const {
		return ((long)(z_end - z_start)) / (long)gridStep;
	}
};

struct FinalExportData {

	const double *altitudes, *temperature, *pressure, *density;
	const int size;

	FinalExportData(const double *altitudes, const double *temperature, const double *pressure, const double *density, const int size)
		: altitudes(altitudes), temperature(temperature), pressure(pressure), density(density), size(size) {
	}

	~FinalExportData() {
		delete[] altitudes;
		delete[] temperature;
		delete[] pressure;
		delete[] density;
	}
};

void exportDataCsv(const FinalExportData &exportData, const string outputFileName) {
	string resultsDir = std::filesystem::current_path().string() + "/results";
	if (!std::filesystem::is_directory(resultsDir) || !std::filesystem::exists(resultsDir)) {
		cout << "Creating results directory at " << resultsDir << "\n";
		std::filesystem::create_directory(resultsDir);
	}

	string outputFilePath = resultsDir + "/" + outputFileName;
	if (outputFilePath.find(".csv") == string::npos) {
		outputFilePath += ".csv";
	}

	cout << "Writing out results to " << outputFilePath << endl;

	ofstream outFile(outputFilePath);
	if (outFile.is_open()) {
		string header = "Altitude (z),Temperature (K),Pressure,Density\n";
#if 0
		cout << header;
#endif
		outFile << header;

		for (auto i = 0; i < exportData.size; i++) {
			double alt = exportData.altitudes[i];
			double temp = exportData.temperature[i];
			double pressure = exportData.pressure[i];
			double density = exportData.density[i];
#if 0
			cout << alt << ","
				<< temp << ","
				<< pressure << ","
				<< density << "\n";
#endif
			outFile << alt << ","
				<< temp << ","
				<< pressure << ","
				<< density << "\n";
		}
		outFile.close();
	} else {
		cout << "Unable to open file\n";
	}
}

__global__ void initAltitudeGrid(const SimProps *p, double *z) {
	const int gid = calc1Dgid();

	if (gid < p->gridSize()) {
		z[gid] = p->gridStep * gid;
	}
}

__global__ void integrate(const SimProps *p, const double *z, double *temperature, double *pressure, double *density) {
	const int gid = calc1Dgid();

	if (gid < p->gridSize()) {
		temperature[gid] = eqhs_phys::temperature(z[gid]);
		pressure[gid] = eqhs_phys::pressure(temperature[gid]);
		density[gid] = eqhs_phys::density(temperature[gid], pressure[gid]);
	}
}

int main() {
	SimProps props;
	const long sizeBytes = props.gridSize() * sizeof(double);

	SimProps *d_props;
	double *d_z;

	double *d_temperature, *d_pressure, *d_density;
	gpuErrCheck(cudaMalloc((void **)&d_temperature, sizeBytes));
	gpuErrCheck(cudaMalloc((void **)&d_pressure, sizeBytes));
	gpuErrCheck(cudaMalloc((void **)&d_density, sizeBytes));

	gpuErrCheck(cudaMalloc((void **)&d_props, sizeof(SimProps)));
	gpuErrCheck(cudaMemcpy(d_props, &props, sizeof(SimProps), cudaMemcpyHostToDevice));
	gpuErrCheck(cudaMalloc((void **)&d_z, sizeBytes));

	dim3 block(WARP_SIZE * 16);
	dim3 grid((props.gridSize() / block.x) + 1);

	initAltitudeGrid << <grid, block >> > (d_props, d_z);
	gpuErrCheck(cudaDeviceSynchronize());

	integrate << <grid, block >> > (d_props, d_z, d_temperature, d_pressure, d_density);
	gpuErrCheck(cudaDeviceSynchronize());

	double *h_temperature = new double[props.gridSize()];
	double *h_pressure = new double[props.gridSize()];
	double *h_density = new double[props.gridSize()];
	double *h_z = new double[props.gridSize()];

	gpuErrCheck(cudaMemcpy((void **)h_z, d_z, sizeBytes, cudaMemcpyDeviceToHost));
	gpuErrCheck(cudaMemcpy((void **)h_temperature, d_temperature, sizeBytes, cudaMemcpyDeviceToHost));
	gpuErrCheck(cudaMemcpy((void **)h_pressure, d_pressure, sizeBytes, cudaMemcpyDeviceToHost));
	gpuErrCheck(cudaMemcpy((void **)h_density, d_density, sizeBytes, cudaMemcpyDeviceToHost));
	FinalExportData exportData(h_z, h_temperature, h_pressure, h_density, props.gridSize());
	exportDataCsv(exportData, "cudaResults.csv");

	gpuErrCheck(cudaFree(d_props));
	gpuErrCheck(cudaFree(d_z));
	gpuErrCheck(cudaFree(d_temperature));
	gpuErrCheck(cudaFree(d_pressure));
	gpuErrCheck(cudaFree(d_density));
	return 0;
}