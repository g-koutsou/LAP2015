#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <sys/time.h>
#include <math.h>
#include <xmmintrin.h>

#define ALIGNMENT 16

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
 * y[0:L-1] <- a*x[0:L-1] + y[0:L-1]
 */
void
axpy(int L, _Complex float a, _Complex float *x, _Complex float *y)
{
  /*
    for(int i=0; i<L; i++)
      y[i] = a*x[i] + y[i];
  */

  /*
    __TODO_3__     

    Add OpenMP pragmas to parallelize the L-loop

    NOTE: Keep va_re, va_im, x0, x1, y0 local, NOT shared
   */
#pragma omp parallel
  {
    float ar = creal(a);
    float ai = cimag(a);
    float aux0[] = { ar, ar, ar, ar};
    float aux1[] = {-ai, ai,-ai, ai};
    register __m128 va_re = _mm_load_ps(aux0);
    register __m128 va_im = _mm_load_ps(aux1);
    register __m128 x0, x1, y0;

#pragma omp for
    for(int i=0; i<L; i+=2) {
      /* 
      
	 __TODO_2__
      
	 Implement the kernel commented above using vector intrinsics

	 Hint - functions to use:
	 _mm_load_ps( );
	 _mm_store_ps( );
	 _mm_mul_ps( );
	 _mm_add_ps( );
	 _mm_shuffle_ps(  ,  , 0b10110001);
      */
      y0 = _mm_load_ps((float *)&y[i]);
      x0 = _mm_load_ps((float *)&x[i]);
      x1 = _mm_shuffle_ps(x0, x0, 0b10110001);
      x1 = _mm_mul_ps(va_im, x1);
      x1 = _mm_add_ps(x1, y0);
      x0 = _mm_mul_ps(va_re, x0);
      y0 = _mm_add_ps(x1, x0);
      _mm_store_ps((float *)&y[i], y0);
    }
  }
  return;
}

/*
 * Fill array of length L with random complex numbers
 */
void
random_vec(int L, _Complex float *v)
{
  for(int i=0; i<L; i++) {
    v[i] = drand48() + _Complex_I*drand48();
  }
  return;
}

/*
 * Usage info
 */
void
usage(char *argv[])
{
  fprintf(stderr, " Usage: %s L nreps\n", argv[0]);
  return;
}

/*
 * Main
 */
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

  _Complex float *x = alloc(sizeof(_Complex float)*L);
  _Complex float *y = alloc(sizeof(_Complex float)*L);
  _Complex float a;

  random_vec(L, x);
  random_vec(L, y);
  random_vec(1, &a);
  
  axpy(L, a, x, y);
  int nreps_inner = 2;
  double tave = 0;
  double tvar = 0;
  for(int k=0; ;k++) {
    tave = 0;
    tvar = 0;
    for(int i=0; i<nreps; i++)
      {
	double t0 = stop_watch(0);
	for(int j=0; j<nreps_inner; j++)
	  axpy(L, a, x, y);
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
     1) Time per kernel call with error (usec)
     2) Susstained floating-point rate (GFlop/sec)
     3) Susstained bandwidth (GBytes/sec)
  */
  double beta_fp = (8*L/tave)*1e-9;
  double beta_io = (8*3*L/tave)*1e-9;
  printf(" L = %12d, %4.2e ± %4.2e usec/call, perf. = %6.4e GFlop/sec, bw = %6.4e GBytes/sec\n",
	 L, tave*1e6, tvar*1e6, beta_fp, beta_io);
  
  free(x);
  free(y);  
  return 0;
}
