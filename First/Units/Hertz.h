#pragma once

namespace Hdd
{
	class Hertz
	{
	public:
		Hertz(int kiloHertz);
		int AsKiloHertz();
		int AsMegaHertz();
		int AsGigaHertz();

	private:
		int kiloHertz_;
	};
}