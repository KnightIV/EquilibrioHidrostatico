
#include "DataUtils.h"

using namespace std;

namespace eqhs_data {

	string get_data_directory() {
		static string dataPath;

		if (dataPath.empty()) {
			cout << "Initializing data path" << endl;

			filesystem::path curPath = filesystem::current_path();
			dataPath = curPath.string() + string("/EQHS_Data/data");
		}

		return dataPath;
	}

	//tuple<double*, int, double*, int> get_alt_temp_values() {
	tuple<shared_ptr<vector<double>>, shared_ptr<vector<double>>> get_alt_temp_values() {
		//static vector<double> altitude, temperature;
		auto altitude = make_shared<vector<double>>();
		auto temperature = make_shared<vector<double>>();

		//if (altitude.empty() && temperature.empty()) {
		//if (altitude->empty() && temperature->empty()) {
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

				altitude->push_back(alt * 1e6); // megameters --> meters
				temperature->push_back(temp);
			}
			dataFile.close();
		}
		//}

		//return { &altitude[0], altitude.size(), &temperature[0], temperature.size() };
		return { altitude, temperature };
	}
}
