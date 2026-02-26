#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <stdbool.h>
#include <math.h>

bool isPrime(long n) {
  if (n <= 1)
    return false;
  if (n == 2)
    return true;
  if (n % 2 == 0) 
    return false;
  int boundary = sqrt(n);
  for (int i = 3; i <= boundary; i += 2)
    if (n % i == 0)
      return false;
  return true;
}

int main(int argc,char **argv) {

  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  //program input argument
  long inputArgument = ins__args.arg; 

  struct timeval ins__tstart, ins__tstop;

  int myrank,nproc;
  
  MPI_Init(&argc,&argv);

  // obtain my rank
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  // and the number of processes
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);

  if(!myrank)
      gettimeofday(&ins__tstart, NULL);


  // run your computations here (including MPI communication)
  long localResult = 0;
  for (int currentNumber = myrank; currentNumber < inputArgument; currentNumber += nproc) {
    if (isPrime(currentNumber))
      localResult++;
  }

  // synchronize/finalize your computations
  long result;
  MPI_Reduce(&localResult, &result, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (!myrank) {
    gettimeofday(&ins__tstop, NULL);
    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
    printf("result=%ld\n", result);
  }
  
  MPI_Finalize();

}
