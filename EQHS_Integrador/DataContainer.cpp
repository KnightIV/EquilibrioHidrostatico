#include "DataContainer.h"

using namespace eqhs_integrador;
using namespace std;

//AltitudeFunction::AltitudeFunction(const double* altitudes, const double* values, const size_t numVals) 
//	: values(values), altitudes(altitudes), numVals(numVals) {}

//AltitudeFunction::AltitudeFunction(const double* altitudes, const double* values, const size_t numVals) : numVals(numVals) {
//	this->altitudes = new double[numVals];
//	this->values = new double[numVals];
//
//	for (int i = 0; i < numVals; i++) {
//		this->altitudes[i] = altitudes[i];
//	}
//}

//AltitudeFunction::AltitudeFunction(const double* altitudes, const double* values, const size_t numVals) : numVals(numVals) {
//	(*this).altitudes(altitudes);
//}

eqhs_integrador::AltitudeFunction::AltitudeFunction(std::shared_ptr<std::vector<double>> altitudes, std::shared_ptr<std::vector<double>> values)
	: altitudes(altitudes), values(values) {
	//if (altitudes->size() != values->size()) {
	//	throw "Incompatible sizes: " + to_string(altitudes->size()) + " != " + to_string(values->size());
	//}
}

size_t eqhs_integrador::AltitudeFunction::size() {
	return altitudes->size();
}

tuple<int, double> eqhs_integrador::AltitudeFunction::find_val_at_z(double z) {
	const double tolerance = 1e-5;

	int minIndex = 0;
	int maxIndex = this->size() - 1;

	while (minIndex < maxIndex) {
		int midPoint = (maxIndex + minIndex) / 2;
		double cur_z = this->altitudes->at(midPoint);

		if (abs(cur_z - z) < tolerance) {
			return { midPoint, this->values->at(midPoint) };
		} else if (cur_z > z) {
			maxIndex = midPoint;
		} else {
			minIndex = midPoint;
		}
	}

	throw "Altitude z = " + to_string(z) + " not found.";
}

tuple<double, double> AltitudeFunction::operator[](int index) {
	return { altitudes->at(index), values->at(index) };
}
