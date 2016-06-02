#include <iostream>
#include "CudaWrapper/CudaWrapper.h"

using std::cout;
using std::endl;

using HddCuda::Device;
using HddCuda::CudaException;

int runTest();

int main(int argc, char** argv)
{
	try
	{
		const auto numDevices = Device::GetNumberOfCudaDevices();
		cout << "CUDA devices: " << numDevices;
		cout << endl;

		for (auto i = 0; i < numDevices; ++i)
		{
			const auto device = Device(i);
			cout << endl;
			device.PrintInfo(cout);
		}

		cout << endl << "Sample code:" << endl;
		runTest();

		return 0;
	}
	catch (const CudaException& e)
	{
		cout << e.what() << endl;
	}
	return 1;
}
