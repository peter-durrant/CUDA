#include "Bytes.h"

namespace Hdd
{
	Bytes::Bytes(size_t bytes) : bytes_(bytes)
	{
	}

	size_t Bytes::AsBytes()
	{
		return bytes_;
	}

	size_t Bytes::AsKiloBytes()
	{
		return bytes_ >> 10;
	}

	size_t Bytes::AsMegaBytes()
	{
		return bytes_ >> 20;
	}

	size_t Bytes::AsGigaBytes()
	{
		return bytes_ >> 30;
	}
}