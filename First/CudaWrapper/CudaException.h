#pragma once

#include <string>
#include <sstream>
#include "cuda_runtime.h"

namespace HddCuda
{
	class CudaException : public std::runtime_error
	{
	public:
		CudaException(cudaError_t error);
		CudaException(const std::string& message);
	};
}
