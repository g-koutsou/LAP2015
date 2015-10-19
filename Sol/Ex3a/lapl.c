#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <complex.h>

#define ND 2
#define ALIGNMENT 16
#define UIDX(v, mu) ((v)*ND + (mu))

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
 * allocates a new field and returns its starting address
 */
void *
new_field(size_t L)
{
  void *ptr = alloc(sizeof(_Complex float)*L*L);
  return ptr;
}

/*
 * allocates a new U(1) gauge field and returns its starting address
 */
void *
new_links(size_t L)
{
  void *ptr = alloc(sizeof(_Complex float)*L*L*ND);
  return ptr;
}

/*
 * Fills u with random entries on the unit circle, gaussianly
 * distributed around 1 + 0*i, using Box-Mueller
 */
void
rand_links(size_t L, _Complex float *u)
{
  for(int i=0; i<L*L; i++) 
    for(int d=0; d<ND; d++) {
      unsigned long int id = d + i*ND;
      double u0 = drand48();
      double u1 = drand48();
      double phi = sqrt(-2*log(u0))*sin(2*M_PI*u1)*M_PI;
      u[id] = cos(phi) + _Complex_I*sin(phi);
    }
  return;
}

/*
 * Fills x with random entries on the unit circle, gaussianly
 * distributed around 1 + 0*i, using Box-Mueller
 */
void
rand_field(size_t L, _Complex float *x)
{
  for(int i=0; i<L*L; i++) {
      double u0 = drand48();
      double u1 = drand48();
      double phi = sqrt(-2*log(u0))*sin(2*M_PI*u1)*M_PI;
      x[i] = cos(phi) + _Complex_I*sin(phi);
  }
  return;
}

/*
 * Fills x with zeros
 */
void
zero_field(size_t L, _Complex float *x)
{
  for(int i=0; i<L*L; i++) {
    x[i] = 0.0;
  }
  return;
}

/*
 * Applies U(1) gauge laplacian to phi_in, with background field u,
 * and returns in phi_out
 */
void
lapl(size_t L, _Complex float *phi_out, _Complex float *phi_in, _Complex float *u)
{
#pragma omp parallel for  
  for(int y=0; y<L; y++)
    for(int x=0; x<L; x++) {
      unsigned long int v00 = y*L + x;
      unsigned long int v0p = y*L + (x+1)%L;
      unsigned long int vp0 = ((y+1)%L)*L + x;
      unsigned long int v0m = y*L + (L+x-1)%L;
      unsigned long int vm0 = ((L+y-1)%L)*L + x;

      _Complex float p;
      p  = phi_in[v0p]*u[UIDX(v00, 0)];
      p += phi_in[vp0]*u[UIDX(v00, 1)];
      p += phi_in[v0m]*conj(u[UIDX(v0m, 0)]);
      p += phi_in[vm0]*conj(u[UIDX(vm0, 1)]);
      phi_out[v00] = 4*phi_in[v00] - p;
    }
      
  return;
}

/*
 * returns y = x^H x for vector x of length L*L
 */
float
xdotx(size_t L, _Complex float *x)
{
  float y = 0; 
  for(int i=0; i<L*L; i++) {
    y += creal(x[i])*creal(x[i]) + cimag(x[i])*cimag(x[i]);
  }
  return y;
}

/*
 * returns z = x^H y for vectors x, y of length L*L
 */
_Complex float
xdoty(size_t L, _Complex float *x, _Complex float *y)
{
  float z = 0; 
  for(int i=0; i<L*L; i++)
    z += conj(x[i])*y[i];
  
  return z;
}

/*
 * returns y = x - y for vectors y, x, of length L*L
 */
void
xmy(size_t L, _Complex float *x, _Complex float *y)
{
  for(int i=0; i<L*L; i++) {
    y[i] = x[i] - y[i];
  }
  return;
}

/*
 * returns y = x for vectors y, x, of length L*L
 */
void
xeqy(size_t L, _Complex float *x, _Complex float *y)
{
  for(int i=0; i<L*L; i++)
    x[i] = y[i];
  
  return;
}

/*
 * returns y = a*x+y for vectors y, x, of length L*L and scalar a
 */
void
axpy(size_t L, _Complex float a, _Complex float *x, _Complex float *y)
{
  for(int i=0; i<L*L; i++)
    y[i] = a*x[i] + y[i];

  return;
}

/*
 * returns y = x+a*y for vectors y, x, of length L*L and scalar a
 */
void
xpay(size_t L, _Complex float *x, _Complex float a, _Complex float *y)
{
  for(int i=0; i<L*L; i++)
    y[i] = x[i] + a*y[i];
  
  return;
}

/*
 * Solves lapl(u) x = b, for x, given b, using Conjugate Gradient
 */
void
cg(size_t L, _Complex float *x, _Complex float *b, _Complex float *u)
{
  int max_iter = 100;
  float tol = 1e-6;

  /* Temporary fields needed for CG */
  _Complex float *r = new_field(L);
  _Complex float *p = new_field(L);
  _Complex float *Ap = new_field(L);

  /* Initial residual and p-vector */
  lapl(L, r, x, u);
  xmy(L, b, r);
  xeqy(L, p, r);

  /* Initial r-norm and b-norm */
  float rr = xdotx(L, r);  
  float bb = xdotx(L, b);
  double t_lapl = 0;
  int iter = 0;
  for(iter=0; iter<max_iter; iter++) {
    printf(" %6d, res = %+e\n", iter, rr/bb);
    if(sqrt(rr/bb) < tol)
      break;
    double t = stop_watch(0);
    lapl(L, Ap, p, u);
    t_lapl += stop_watch(t);
    float pAp = xdoty(L, p, Ap);
    float alpha = rr/pAp;
    axpy(L, alpha, p, x);
    axpy(L, -alpha, Ap, r);
    float r1r1 = xdotx(L, r);
    float beta = r1r1/rr;
    xpay(L, r, beta, p);
    rr = r1r1;
  }

  /* Recompute residual after convergence */
  lapl(L, r, x, u);
  xmy(L, b, r);
  rr = xdotx(L, r);

  double beta_fp = 34*L*L/(t_lapl/(double)iter)*1e-9;
  double beta_io = 32*L*L/(t_lapl/(double)iter)*1e-9;
  printf(" Converged after %6d iterations, res = %+e\n", iter, rr/bb);  
  printf(" Time in lapl(): %+6.3e sec/call, %4.2e Gflop/s, %4.2e GB/s\n",
	 t_lapl/(double)iter, beta_fp, beta_io);  

  free(r);
  free(p);
  free(Ap);
  return;
}

/*
 * Usage info
 */
void
usage(char *argv[])
{
  fprintf(stderr, " Usage: %s L\n", argv[0]);
  return;
}

int
main(int argc, char *argv[])
{
  if(argc != 2) {
    usage(argv);
    exit(1);
  }

  char *e;
  size_t L = (int)strtoul(argv[1], &e, 10);
  if(*e != '\0') {
    usage(argv);
    exit(2);
  }

  _Complex float *b = new_field(L);
  _Complex float *x = new_field(L);
  _Complex float *u = new_links(L);

  rand_links(L, u);
  rand_field(L, b);
  zero_field(L, x);

  cg(L, x, b, u);
  
  free(b);
  free(x);
  free(u);
  return 0;
}
