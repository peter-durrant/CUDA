#include <iostream>
#include "cudadevices.h"
#include "cudaexception.h"

using std::cout;
using std::endl;

int runTest();

int main(int argc, char** argv)
{
	try
	{
		const auto numDevices = HddCuda::Device::GetNumberOfCudaDevices();
		cout << "CUDA devices: " << numDevices;
		cout << endl;

		for (auto i = 0; i < numDevices; ++i)
		{
			const auto device = HddCuda::Device(i);
			cout << endl;
			device.PrintInfo(cout);
		}

		cout << endl << "Sample code:" << endl;
		runTest();

		return 0;
	}
	catch (const HddCuda::CudaException& e)
	{
		cout << e.what() << endl;
	}
	return 1;
}
