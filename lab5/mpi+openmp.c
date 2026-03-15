#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include <stdbool.h>

const int TAG_RESULT = 0;

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

long countTwinPrimes(long start, long end, long limit)
{
  long count = 0;
  if (start % 2 == 0)
    start++;
  bool prevIsPrime = false;
  for (long i = start; i < end; i += 2)
  {
    if (isPrime(i))
    {
      if (prevIsPrime)
        count++;
      prevIsPrime = true;
    }
    else
    {
      prevIsPrime = false;
    }
  }

  if (prevIsPrime)
  {
    if (end % 2 == 0)
      end++;
    if (end <= limit && isPrime(end))
      count++;
  }
  return count;
}

void work(int rank, long start, long end, long limit)
{
  const long stepSize = 1000;

  long result = 0, localResult = 0;
#pragma omp parallel for private(localResult) reduction(+ : result)
  for (long i = start; i < end; i += stepSize)
  {
    long localStart = i;
    long localEnd = (i + stepSize < end) ? i + stepSize : end;
    localResult = countTwinPrimes(localStart, localEnd, limit);
    result += localResult;
    printf("Rank %d, thread %d: Counted %ld twin primes in range [%ld, %ld)\n", rank, omp_get_thread_num(), localResult, localStart, localEnd);
  }

  MPI_Send(&result, 1, MPI_LONG, 0, TAG_RESULT, MPI_COMM_WORLD);
}

int main(int argc, char **argv)
{

  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  // set number of threads
  omp_set_num_threads(ins__args.n_thr);

  // program input argument
  long inputArgument = ins__args.arg;

  struct timeval ins__tstart, ins__tstop;

  int threadsupport;
  int myrank, nproc;
  // Initialize MPI with desired support for multithreading -- state your desired support level

  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &threadsupport);

  if (threadsupport < MPI_THREAD_FUNNELED)
  {
    printf("\nThe implementation does not support MPI_THREAD_FUNNELED, it supports level %d\n", threadsupport);
    MPI_Finalize();
    return -1;
  }

  // obtain my rank
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  // and the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (!myrank)
  {
    gettimeofday(&ins__tstart, NULL);
  }
  // run your computations here (including MPI communication and OpenMP stuff)

  long stepSize = inputArgument / nproc;
  long start = myrank * stepSize;
  long end = (myrank == nproc - 1) ? inputArgument : start + stepSize;

  work(myrank, start, end, inputArgument);

  // synchronize/finalize your computations
  if (!myrank)
  {
    long resultTemp, result = 0;
    for (int i = 0; i < nproc; i++)
    {
      MPI_Recv(&resultTemp, 1, MPI_LONG, i, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      result += resultTemp;
    }
    printf("Twin primes up to %ld: %ld\n", inputArgument, result);
  }

  if (!myrank)
  {
    gettimeofday(&ins__tstop, NULL);
    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
  }

  MPI_Finalize();
}
