#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

const int threadsInBlock = 1024;

__device__
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
void work(int *result, long limit) {
	long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (isPrime(idx) && isPrime(idx + 2) && idx + 2 <= limit) {
    result[idx] = 1;
  } else {
    result[idx] = 0;
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

	int *hostResults = (int*) malloc(size * sizeof(int));
	if (!hostResults) {
		printf("Error allocating memory on the host\n");
		exit(EXIT_FAILURE);
	}
	int *deviceResults;
	if (cudaSuccess != cudaMalloc((void **)&deviceResults, size * sizeof(int))) {
		printf("Error allocating memory on the GPU\n");
		printCudaError();
		free(hostResults);
	}

	work<<<blockCount, threadsInBlock>>>(deviceResults, inputArgument);
	if (cudaSuccess != cudaGetLastError()) {
		printf("Error during kernel launch\n");
		printCudaError();
		free(hostResults);
		cudaFree(deviceResults);
		exit(EXIT_FAILURE);
	}

	if (cudaSuccess != cudaMemcpy(hostResults, deviceResults, size * sizeof(int), cudaMemcpyDeviceToHost)) {
		printf("Error copying results\n");
		printCudaError();
		free(hostResults);
		cudaFree(deviceResults);
		exit(EXIT_FAILURE);
	}

	// synchronize/finalize your CUDA computations
	long result = 0;
	for (int i = 0; i < size; i++) {
		result += hostResults[i];
	}
	printf("Twin primes up to %ld: %ld\n", inputArgument, result);


	gettimeofday(&ins__tstop, NULL);
	ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);

	// free any allocated memory
	free(hostResults);
	cudaFree(deviceResults);
}
