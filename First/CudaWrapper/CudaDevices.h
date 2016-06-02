#pragma once

#include <string>
#include "cuda_runtime.h"
#include "Units/Units.h"

namespace HddCuda
{
	class Device
	{
	public:
		static int GetNumberOfCudaDevices();

		Device(int device);
		std::string Name() const;

		int NumberOfMultiprocessors() const;
		int NumberOfCudaCoresPerMultiprocessor() const;
		int NumberOfCudaCores() const;

		Hdd::Hertz ClockRate() const;
		Hdd::Bytes Memory() const;

		void PrintInfo(std::ostream& os) const;

	private:
		const cudaDeviceProp properties_;
		const int deviceIndex_;
	};
}
