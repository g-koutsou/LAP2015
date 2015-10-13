#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifndef N
#define N 1
#endif

/* 
   Convenience indexing macro 
*/
#define NIJ(I, J) (I)*N + (J)

#define ALIGNMENT 16

/*
 * Returns seconds elapsed since t0
 */
double
stop_watch(double t0)
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  double t1 = tp.tv_sec + tp.tv_usec*1e-6;  
  return t1-t0;
}

/*
 * malloc with minimal error detection
 */
void *
alloc(size_t size)
{
  void *ptr;
  posix_memalign(&ptr, ALIGNMENT, size);
  if(ptr == NULL) {
    fprintf(stderr, " malloc() returned NULL. Out of memory?\n");
    exit(-1);
  }
  return ptr;
}

/*
 * Random vector of N*N elements
 */
void
randNxN(double *p)
{
  for(int i=0; i<N*N; i++)
    p[i] = drand48();
  return;
}

/*
 * y^{ab}[0:(L-1)] <- a^{ac} * x^{cb}[0:(L-1)]
 */
void
mulNxN(int L, double *y, double *a, double *x)
{
#pragma omp parallel for
  for(int il=0; il<L; il++) {
    double *yl = &y[il*N*N];
    double *xl = &x[il*N*N];
    for(int i=0; i<N; i++) {
      for(int j=0; j<N; j++) {
	yl[NIJ(i,j)] = 0;
	for(int k=0; k<N; k++) {
	  yl[NIJ(i,j)] += a[NIJ(i,k)]*xl[NIJ(k,j)];
	}
      }
    }
  }
  return;
}

void
usage(char *argv[])
{
  fprintf(stderr, " Usage: %s L nreps\n", argv[0]);
  return;
}

int
main(int argc, char *argv[])
{
  if(argc != 3) {
    usage(argv);
    exit(1);
  }

  char *e;
  int L = (int)strtoul(argv[1], &e, 10);
  if(*e != '\0') {
    usage(argv);
    exit(2);
  }

  int nreps = (int)strtoul(argv[2], &e, 10);
  if(*e != '\0') {
    usage(argv);
    exit(2);
  }

  double *x = alloc(sizeof(double)*L*N*N);
  double *y = alloc(sizeof(double)*L*N*N);
  double *a = alloc(sizeof(double)*N*N);

  randNxN(a);
  for(int i=0; i<L; i++)
    randNxN(&y[i]);
  for(int i=0; i<L; i++)
    randNxN(&x[i]);

  int nreps_inner = 2;
  double tave = 0;
  double tvar = 0;
  for(int k=0; ;k++) {
    tave = 0;
    tvar = 0;
    mulNxN(L, y, a, x);    
    for(int i=0; i<nreps; i++)
      {
	double t0 = stop_watch(0);
	for(int j=0; j<nreps_inner; j++)
	  mulNxN(L, y, a, x);
	t0 = stop_watch(t0)/nreps_inner;
	tave += t0;
	tvar += t0*t0;
      }
    tave /= (double)nreps;
    tvar /= (double)nreps;
    tvar = sqrt(tvar - tave*tave);
    if(tvar < tave/25)
      break;
    nreps_inner = nreps_inner*2;    
  }

  /*
    ___TODO_1___

    Print:
     1) Time per kernel call
     2) Susstained floating-point rate (GFlop/sec)
     3) Susstained bandwidth (GBytes/sec)
     Note: keep as function of N
   */
  double beta_fp = (2*N-1)*N*N*L/tave*1e-9;
  double beta_io = (2*N*N)*L/tave*1e-9;
  printf(" N = %2d, L = %12d, %4.2e ± %4.2e usec/call, perf. = %6.4e GFlop/sec, bw = %6.4e GBytes/sec\n",
	 N, L, tave*1e6, tvar*1e6, beta_fp, beta_io);
  free(x);
  free(y);  
  return 0;
}
