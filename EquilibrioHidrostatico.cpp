// EquilibrioHidrostatico.cpp : Defines the entry point for the application.

#include "EquilibrioHidrostatico.h"

using namespace std;
using namespace eqhs_integrador;

int main() {
	auto start = chrono::system_clock::now();
	auto tup = eqhs_data::get_alt_temp_values();
	shared_ptr<vector<double>> altitudes = get<0>(tup);
	shared_ptr<vector<double>> temperatures = get<1>(tup);

	AltitudeFunction temp_altitude = AltitudeFunction(altitudes, temperatures);

	auto fluid_functions = eqhs_integrador::equilibrium_equations_values(temp_altitude);
	AltitudeFunction scale_height_altitude = get<0>(fluid_functions);
	AltitudeFunction pressure_altitude = get<1>(fluid_functions);
	AltitudeFunction density_altitude = get<2>(fluid_functions);

	eqhs_data::FinalExportData exportData(altitudes, temperatures,
		pressure_altitude.values, density_altitude.values,
		scale_height_altitude.values);
	eqhs_data::export_data_csv(exportData, "results");

	auto end = chrono::system_clock::now();
	chrono::duration<double> elapsed_seconds = end - start;
	time_t end_time = chrono::system_clock::to_time_t(end);
	cout << "\nFinished computation at " << ctime(&end_time) << "Elapsed time: " << elapsed_seconds.count() << "s" << endl;
	return 0;
}