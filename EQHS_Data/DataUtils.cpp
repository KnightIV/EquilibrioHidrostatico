
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

	tuple<shared_ptr<vector<double>>, shared_ptr<vector<double>>> get_alt_temp_values() {
		auto altitude = make_shared<vector<double>>();
		auto temperature = make_shared<vector<double>>();

		string dataFilePath = get_data_directory() + "/" + TEMPERATURE_ALTITUDE_FILE_NAME;
		cout << "Reading in temperature-altitude values from " << dataFilePath << endl;

		fstream dataFile(dataFilePath, ios::in);
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

		return { altitude, temperature };
	}

	void export_data_csv(const FinalExportData& exportData, const string outputFileName) {
		string resultsDir = filesystem::current_path().string() + "/results";
		if (!filesystem::is_directory(resultsDir) || !filesystem::exists(resultsDir)) { 
			cout << "Creating results directory at " << resultsDir << "\n";
			filesystem::create_directory(resultsDir);
		}

		string outputFilePath = resultsDir + "/" + outputFileName;
		if (outputFilePath.find(".csv") == string::npos) {
			outputFilePath += ".csv";
		}

		cout << "Writing out results to " << outputFilePath << endl;

		ofstream outFile(outputFilePath);
		if (outFile.is_open()) {
			string header = "Altitude (z),Temperature (K),Scale Height,Pressure,Density\n";
			cout << header;
			outFile << header;

			for (auto i = 0; i < exportData.size(); i++) {
				double alt = exportData.altitudes->at(i);
				double temp = exportData.temperature->at(i);
				double scaleHeight = exportData.scaleHeight->at(i);
				double pressure = exportData.pressure->at(i);
				double density = exportData.density->at(i);
				cout << alt << ","
					<< temp << ","
					<< scaleHeight << ","
					<< pressure << ","
					<< density << "\n";
				outFile << alt << ","
					<< temp << ","
					<< scaleHeight << ","
					<< pressure << ","
					<< density << "\n";
			}
			outFile.close();
		} else {
			cout << "Unable to open file\n";
		}
	}

	FinalExportData::FinalExportData(std::shared_ptr<std::vector<double>> altitudes,
		std::shared_ptr<std::vector<double>> temperature,
		std::shared_ptr<std::vector<double>> pressure,
		std::shared_ptr<std::vector<double>> density,
		std::shared_ptr<std::vector<double>> scaleHeight)
		: altitudes(altitudes), temperature(temperature), pressure(pressure), density(density), scaleHeight(scaleHeight) {}

	size_t FinalExportData::size() const {
		return altitudes->size();
	}
}
