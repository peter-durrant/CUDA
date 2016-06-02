#pragma once

namespace Hdd
{
	class Bytes
	{
	public:
		Bytes(size_t bytes);

		size_t AsBytes();
		size_t AsKiloBytes();
		size_t AsMegaBytes();
		size_t AsGigaBytes();

	private:
		size_t bytes_;
	};
}