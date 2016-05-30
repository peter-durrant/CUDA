#include "cudaexception.h"
#include <string>
#include <sstream>

namespace HddCuda
{
	std::string GetErrorMessage(cudaError_t error)
	{
		std::stringstream ss;
		ss << "CUDA error: " << error;
		return ss.str();
	}

	CudaException::CudaException(cudaError_t error) :
		runtime_error(GetErrorMessage(error))
	{
	}

	CudaException::CudaException(const std::string& message) :
		runtime_error(message)
	{
	}
}
