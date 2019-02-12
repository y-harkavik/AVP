#include <stdio.h>
#include <conio.h>
#include <intrin.h>
#include <iostream>
#pragma intrinsic(__rdtsc)

using namespace std;

#define BLOCK_SIZE  1024 * 32 // 32 КБ размер кэша
#define OFFSET 1024 * 1024 // 1 МБ расстояние между блоками
#define N 20

struct CacheString
{
	int bytes[15];
	int next;
};

void fill_block(CacheString *block, int n);

int main()
{
	CacheString *block = new CacheString[327680];
	long long time[N];

	for (int i = 1; i <= N; i++) {
		register unsigned long long begin = 0.0, end = 0.0;
		register int index = 0;

		fill_block(block, i);

		begin = __rdtsc();

		for (int j = 0; j < 1000000; j++) {
			do
			{
				index = block[index].next;
			} while (index != 0);
		}

		end = __rdtsc();


		time[i - 1] = end - begin;
	}

	for (int i = 0; i < N; i++)
	{
		cout << i + 1 << ": " << time[i] / 10000000 << endl;
	}

	_getch();

	return 0;
}

void fill_block(CacheString *block, int n)
{
	for (int i = 0; i < n; i++)
	{
		if (i + 1 == n)
		{
			for (int j = 0; j < BLOCK_SIZE / (n * 64); j++)
			{

				if ((j + 1) == BLOCK_SIZE / (n * 64))
				{
					block[i * OFFSET / 64 + j].next = 0;
					return;
				}

				block[i * OFFSET / 64 + j].next = j + 1;
			}
		}
		for (int j = 0; j < BLOCK_SIZE / (n * 64); j++)
		{
			block[i * OFFSET / 64 + j].next = i * OFFSET / 64 + OFFSET / 64 + j;
		}
	}
}
