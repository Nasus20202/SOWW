#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

const int threadsInBlock = 1024;

bool isPrime(long n)
{
	if (n <= 1)
		return false;
	if (n == 2)
		return true;
	if (n % 2 == 0)
		return false;
	for (long i = 3; i * i <= n; i += 2)
		if (n % i == 0)
			return false;
	return true;
}

__global__
void sieveKernel(int *sieve, long size, long *basePrimes, long baseCount) {
	long idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < baseCount) {
		long prime = basePrimes[idx];
		for (long j = prime * prime; j < size; j += prime) {
			sieve[j] = 0; // mark as non-prime
		}
	}
}

__global__
void countTwinPrimesKernel(int *sieve, long size, unsigned long long *twinCount) {
	long idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 3 && idx <= size - 2) {
		if (sieve[idx] && sieve[idx + 2]) {
			atomicAdd(twinCount, 1ULL);
		}
	}
}

void printCudaError() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
	}
}

int main(int argc,char **argv) {
	Args ins__args;
	parseArgs(&ins__args, &argc, argv);

	//program input argument
	long inputArgument = ins__args.arg;

	struct timeval ins__tstart, ins__tstop;
	gettimeofday(&ins__tstart, NULL);

	// run your CUDA kernel(s) here
	long blockCount = inputArgument / threadsInBlock;
	if (blockCount * threadsInBlock < inputArgument) {
		blockCount++; // round up to the next block size
	}
	long size = blockCount * threadsInBlock;
	printf("Using %ld blocks and %d threads per block, total of %ld threads\n", blockCount, threadsInBlock, size);

	// calculate primes up to sqrt(inputArgument) to optimize the twin prime checking
	long sqrtLimit = (long)sqrt((double)inputArgument) + 1;
	long *basePrimes = (long*)calloc(sqrtLimit, sizeof(long));
	long count = 0;
	for (long i = 2; i < sqrtLimit; i++) {
		if (isPrime(i)) {
			basePrimes[count++] = i;
		}
	}
	long *deviceBasePrimes;
	if (cudaMalloc(&deviceBasePrimes, count * sizeof(long)) != cudaSuccess) {
		printCudaError();
		return EXIT_FAILURE;
	}
	if (cudaMemcpy(deviceBasePrimes, basePrimes, count * sizeof(long), cudaMemcpyHostToDevice) != cudaSuccess) {
		printCudaError();
		return EXIT_FAILURE;
	}

	// prepare prime sieve on the device
	int *deviceSieve;
	if (cudaMalloc(&deviceSieve, size * sizeof(int)) != cudaSuccess) {
		printCudaError();
		return EXIT_FAILURE;
	}
	if (cudaMemset(deviceSieve, 1, size * sizeof(int)) != cudaSuccess) {
		printCudaError();
		return EXIT_FAILURE;
	}
	sieveKernel<<<blockCount, threadsInBlock>>>(deviceSieve, size, deviceBasePrimes, count);
	if (cudaGetLastError() != cudaSuccess) {
		printCudaError();
		return EXIT_FAILURE;
	}

	// synchronize/finalize your CUDA computations
	// calculate twin primes on the device with the sieve results
	unsigned long long *deviceTwinCount;
	if (cudaMalloc(&deviceTwinCount, sizeof(unsigned long long)) != cudaSuccess) {
		printCudaError();
		return EXIT_FAILURE;
	}
	if (cudaMemset(deviceTwinCount, 0, sizeof(unsigned long long)) != cudaSuccess) {
		printCudaError();
		return EXIT_FAILURE;
	}
	countTwinPrimesKernel<<<blockCount, threadsInBlock>>>(deviceSieve, inputArgument, deviceTwinCount);
	if (cudaGetLastError() != cudaSuccess) {
		printCudaError();
		return EXIT_FAILURE;
	}

	unsigned long long result;
	if (cudaMemcpy(&result, deviceTwinCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) {
		printCudaError();
		return EXIT_FAILURE;
	}
	printf("Twin primes up to %ld: %llu\n", inputArgument, result);


	gettimeofday(&ins__tstop, NULL);
	ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);

	// free any allocated memory
	free(basePrimes);
	cudaFree(deviceBasePrimes);
	cudaFree(deviceSieve);
  cudaFree(deviceTwinCount);
}
