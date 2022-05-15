// EquilibrioHidrostatico.cpp : Defines the entry point for the application.
//

#include "EquilibrioHidrostatico.h"

using namespace std;

int main() {
	auto start = chrono::system_clock::now();
	const auto [altitudes, alts_size, temperatures, temps_size] = eqhs_data::get_alt_temp_values();

	for (int i = 0; i < alts_size; i++) {
		cout << "(" << altitudes[i] << ", " << temperatures[i] << ")" << endl;
	}

	auto end = chrono::system_clock::now();
	chrono::duration<double> elapsed_seconds = end - start;
	time_t end_time = chrono::system_clock::to_time_t(end);
	cout << "Finished computation at " << ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << "s" << endl;
	return 0;
}