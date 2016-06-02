#include "Hertz.h"

namespace Hdd
{
	Hertz::Hertz(int hertz) : kiloHertz_(hertz)
	{
	}

	int Hertz::AsKiloHertz()
	{
		return kiloHertz_;
	}

	int Hertz::AsMegaHertz()
	{
		return kiloHertz_ / 1000;
	}

	int Hertz::AsGigaHertz()
	{
		return kiloHertz_ / 1000000;
	}
}