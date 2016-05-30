#include "cudadevices.h"
#include "cudaexception.h"

namespace HddCuda
{
	namespace
	{
		void HandleError(const cudaError_t error)
		{
			if (error != cudaSuccess)
			{
				throw new CudaException(error);
			}
		}

		cudaDeviceProp GetCudaDeviceProperties(int device)
		{
			cudaDeviceProp properties;
			const cudaError_t error = cudaGetDeviceProperties(&properties, device);
			HandleError(error);
			return properties;
		}
	}

	size_t Device::AsKiloBytes(size_t bytes)
	{
		return bytes >> 10;
	}

	size_t Device::AsMegaBytes(size_t bytes)
	{
		return bytes >> 20;
	}

	size_t Device::AsGigaBytes(size_t bytes)
	{
		return bytes >> 30;
	}

	int Device::AsMegaHertz(int kiloHertz)
	{
		return kiloHertz / 1000;
	}

	int Device::AsGigaHertz(int kiloHertz)
	{
		return kiloHertz / 1000 / 1000;
	}

	int Device::GetNumberOfCudaDevices()
	{
		int numDevices;
		const cudaError_t error = cudaGetDeviceCount(&numDevices);
		HandleError(error);
		return numDevices;
	}

	Device::Device(int device) :
		properties_(GetCudaDeviceProperties(device)),
		deviceIndex_(device)
	{
	}

	std::string Device::Name() const
	{
		return properties_.name;
	}

	int Device::NumberOfMultiprocessors() const
	{
		return properties_.multiProcessorCount;
	}

	int Device::NumberOfCudaCoresPerMultiprocessor() const
	{
		int coresPerMultiprocessor = 0;
		switch (properties_.major)
		{
		case 1: // 1.x
			coresPerMultiprocessor = 8;
			break;
		case 2:
			if (properties_.minor == 0)
			{
				// 2.0
				coresPerMultiprocessor = 32;
			}
			else if (properties_.minor == 1)
			{
				// 2.1
				coresPerMultiprocessor = 48;
			}
			break;
		case 3:
			// 3.x
			coresPerMultiprocessor = 192;
			break;
		case 5:
			// 5.x
			coresPerMultiprocessor = 128;
			break;
		}
		if (coresPerMultiprocessor > 0)
		{
			return coresPerMultiprocessor;
		}
		throw new CudaException([](const cudaDeviceProp& properties)
		{
			std::stringstream ss;
			ss << "Unrecognised compute capability: " << properties.major << "." << properties.minor;
			return ss.str();
		}(properties_));
	}

	int Device::NumberOfCudaCores() const
	{
		const int coresPerMultiprocessor = NumberOfCudaCoresPerMultiprocessor();
		return properties_.multiProcessorCount * coresPerMultiprocessor;
	}

	void Device::PrintInfo(std::ostream& os) const
	{
		os << "Device:                        " << deviceIndex_ << std::endl;
		os << "Name:                          " << properties_.name << std::endl;
		os << "Compute capability:            " << properties_.major << "." << properties_.minor << std::endl;
		os << "Total global memory:           " << AsGigaBytes(properties_.totalGlobalMem) << " GiB (" << properties_.totalGlobalMem << " bytes)" << std::endl;
		os << "Clock speed:                   " << AsMegaHertz(properties_.clockRate) << " MHz (" << properties_.clockRate << " kHz)" << std::endl;
		os << "Streaming multiprocessors:     " << properties_.multiProcessorCount << std::endl;
		os << "CUDA cores per multiprocessor: " << NumberOfCudaCoresPerMultiprocessor() << std::endl;
		os << "CUDA cores:                    " << NumberOfCudaCores() << std::endl;
	}
}
