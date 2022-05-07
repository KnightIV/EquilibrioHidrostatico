
#include "DataUtils.h"

using namespace std;

namespace eqhs {

	const string TEMPERATURE_ALTITUDE_FILE_NAME = "Temperature-C7.dat";

	string get_data_directory() {
		static string dataPath;

		if (dataPath.empty()) {
			cout << "Initializing data path" << endl;

			char cwd[256];
			getcwd(cwd, sizeof(cwd) / sizeof(char));
			dataPath = cwd + string("/data");
		}

		return dataPath;
	}

	tuple<double*, int, double*, int> get_alt_temp_values() {
		static vector<double> altitude, temperature;
		
		if (altitude.empty() && temperature.empty()) {
			string dataFilePath = get_data_directory() + "/" + TEMPERATURE_ALTITUDE_FILE_NAME;
			cout << "Reading in temperature-altitude values from " << dataFilePath << endl;

			fstream dataFile;

			dataFile.open(dataFilePath, ios::in); 
			if (dataFile.is_open()) {
				string line;
				while (getline(dataFile, line)) {
					double alt, temp;
					stringstream ss(line);
					ss >> alt >> temp;

					altitude.push_back(alt);
					temperature.push_back(temp);
				}
				dataFile.close();
			}
		}

		return { &altitude[0], altitude.size(), &temperature[0], temperature.size() };
	}
}
