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
	if (altitudes->size() != values->size()) {
		throw "Incompatible sizes: " + to_string(altitudes->size()) + " != " + to_string(values->size());
	}
}

size_t eqhs_integrador::AltitudeFunction::size() {
	return altitudes->size();
}

tuple<double, double> AltitudeFunction::operator[](int index) {
	return { altitudes->at(index), values->at(index) };
}
