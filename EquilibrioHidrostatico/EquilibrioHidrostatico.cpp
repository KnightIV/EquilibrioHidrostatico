// EquilibrioHidrostatico.cpp : Defines the entry point for the application.
//

#include "EquilibrioHidrostatico.h"

using namespace std;

int main() {
	auto start = chrono::system_clock::now();

	double* altitudes = get<0>(eqhs::get_alt_temp_values());
	size_t alts_size = get<1>(eqhs::get_alt_temp_values());

	double* temperatures = get<2>(eqhs::get_alt_temp_values());
	size_t temps_size = get<3>(eqhs::get_alt_temp_values());

	auto end = chrono::system_clock::now();
	chrono::duration<double> elapsed_seconds = end - start;
	time_t end_time = chrono::system_clock::to_time_t(end);
	cout << "Finished computation at " << ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << "s" << endl;
	return 0;
}