#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <stdbool.h>

const long stepSize = 1000;

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

int main(int argc, char **argv)
{

  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  // set number of threads
  omp_set_num_threads(ins__args.n_thr);

  // program input argument
  long inputArgument = ins__args.arg;

  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);

  // run your computations here (including OpenMP stuff)
  long globalResult = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : globalResult)
  for (long start = 0; start < inputArgument; start += stepSize)
  {
    long localEnd = start + stepSize;
    if (localEnd > inputArgument)
      localEnd = inputArgument;

    long x = countTwinPrimes(start, localEnd, inputArgument);
    printf("Thread %d found %ld twin primes in range [%ld, %ld)\n", omp_get_thread_num(), x, start, localEnd);
    globalResult += x;
  }

  printf("Twin primes up to %ld: %ld\n", inputArgument, globalResult);

  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
}
