// EquilibrioHidrostatico.cpp : Defines the entry point for the application.
//

#include "EquilibrioHidrostatico.h"

using namespace std;
using namespace eqhs_integrador;

int main() {
	auto start = chrono::system_clock::now();
	//const auto [altitudes, alts_size, temperatures, temps_size] = eqhs_data::get_alt_temp_values();
	//auto temp_altitude = eqhs_integrador::AltitudeFunction(altitudes, temperatures, temps_size);
	//auto [altitudes, temperatures] = eqhs_data::get_alt_temp_values();
	tuple<shared_ptr<vector<double>>, shared_ptr<vector<double>>> tup = eqhs_data::get_alt_temp_values();
	shared_ptr<vector<double>> altitudes = get<0>(tup);
	shared_ptr<vector<double>> temperatures = get<1>(tup);

	eqhs_integrador::AltitudeFunction temp_altitude = eqhs_integrador::AltitudeFunction(altitudes, temperatures);
	eqhs_integrador::AltitudeFunction scale_height_altitude = eqhs_integrador::scale_height(temp_altitude);

	cout << "\n";
	for (int i = 0; i < temp_altitude.size(); i++) {
		auto tempTup = temp_altitude[i];
		double alt = get<0>(tempTup);
		double temp = get<1>(tempTup);
		
		double scaleHeight = get<1>(scale_height_altitude[i]);

		cout << "[" << alt << ", " << temp <<  ", " << scaleHeight << "]\n";
	}

	auto end = chrono::system_clock::now();
	chrono::duration<double> elapsed_seconds = end - start;
	time_t end_time = chrono::system_clock::to_time_t(end);
	cout << "\nFinished computation at " << ctime(&end_time) << "Elapsed time: " << elapsed_seconds.count() << "s" << endl;
	return 0;
}