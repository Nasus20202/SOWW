#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <stdbool.h>

#define RANGE_SIZE 1000
#define TAG_RANGE 0
#define TAG_RESULT 1
#define TAG_FINISH 2

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
    long next = (end % 2 == 0) ? end + 1 : end;
    if (next <= limit && isPrime(next))
      count++;
  }
  return count;
}

#pragma region Master
void shutdownSlave(int rank)
{
  printf("Shutting down process %d\n", rank);
  MPI_Send(NULL, 0, MPI_LONG, rank, TAG_FINISH, MPI_COMM_WORLD);
}

void sendRange(int rank, long range[2])
{
  printf("Sending range [%ld, %ld] to process %d\n", range[0], range[1], rank);
  MPI_Send(range, 2, MPI_LONG, rank, TAG_RANGE, MPI_COMM_WORLD);
}

void receiveResult(int rank, long *result, MPI_Status *status)
{
  MPI_Recv(result, 1, MPI_LONG, rank, TAG_RESULT, MPI_COMM_WORLD, status);
  printf("Received result %ld from process %d\n", *result, rank);
}
#pragma endregion

#pragma region Slave
void probeMaster(MPI_Status *status)
{
  MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, status);
}

void receiveRange(long range[2])
{
  MPI_Recv(range, 2, MPI_LONG, 0, TAG_RANGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void sendResult(long result)
{
  MPI_Send(&result, 1, MPI_LONG, 0, TAG_RESULT, MPI_COMM_WORLD);
}

void receiveFinish()
{
  MPI_Recv(NULL, 0, MPI_LONG, 0, TAG_FINISH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
#pragma endregion

void master(long arraySize, int procCount)
{
  long range[2] = {1, 0};
  long result = 0;
  int activeSlaves = procCount - 1;
  // Split initial tasks to workers
  for (int i = 1; i < procCount; i++)
  {
    if (range[0] >= arraySize)
    {
      shutdownSlave(i);
      activeSlaves--;
      continue;
    }
    range[1] = range[0] + RANGE_SIZE;
    if (range[1] > arraySize)
      range[1] = arraySize;
    sendRange(i, range);
    range[0] = range[1];
  }

  // Then split remaining tasks to workers as they finish their work
  while (range[1] < arraySize)
  {
    long resultTemp;
    MPI_Status status;
    receiveResult(MPI_ANY_SOURCE, &resultTemp, &status);
    result += resultTemp;

    range[1] = range[0] + RANGE_SIZE;
    if (range[1] > arraySize)
      range[1] = arraySize;

    sendRange(status.MPI_SOURCE, range);
    range[0] = range[1];
  }

  // Receive results from remaining active slaves and shut them down
  for (int i = 1; i <= activeSlaves; i++)
  {
    long resultTemp;
    MPI_Status status;
    receiveResult(MPI_ANY_SOURCE, &resultTemp, &status);
    result += resultTemp;
    shutdownSlave(status.MPI_SOURCE);
  }

  printf("Twin primes up to %ld: %ld\n", arraySize, result);
}

void slave(long arraySize, int rank)
{
  MPI_Status status;
  while (true)
  {
    probeMaster(&status);
    if (status.MPI_TAG == TAG_FINISH)
    {
      receiveFinish();
      break;
    }
    if (status.MPI_TAG == TAG_RANGE)
    {
      long range[2];
      receiveRange(range);
      long result = countTwinPrimes(range[0], range[1], arraySize);
      sendResult(result);
    }
  }
}

int main(int argc, char **argv)
{
  Args ins__args;
  parseArgs(&ins__args, &argc, argv);
  long inputArgument = ins__args.arg;

  struct timeval ins__tstart, ins__tstop;
  int rank, procCount;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procCount);

  if (rank == 0)
  {
    gettimeofday(&ins__tstart, NULL);

    master(inputArgument, procCount);

    gettimeofday(&ins__tstop, NULL);
    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
  }
  else
  {
    slave(inputArgument, rank);
  }

  MPI_Finalize();
}
