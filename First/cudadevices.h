#include <string>
#include "cuda_runtime.h"

namespace HddCuda
{
	class Device
	{
	public:
		static int GetNumberOfCudaDevices();

		static size_t AsKiloBytes(size_t bytes);
		static size_t AsMegaBytes(size_t bytes);
		static size_t AsGigaBytes(size_t bytes);

		static int AsMegaHertz(int kiloHertz);
		static int AsGigaHertz(int kiloHertz);

		Device(int device);
		std::string Name() const;

		int NumberOfMultiprocessors() const;
		int NumberOfCudaCoresPerMultiprocessor() const;
		int NumberOfCudaCores() const;

		void PrintInfo(std::ostream& os) const;

	private:
		const cudaDeviceProp properties_;
		const int deviceIndex_;
	};
}
