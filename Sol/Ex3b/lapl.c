#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <complex.h>
#include <xmmintrin.h>

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
new_field(int L)
{
  void *ptr = alloc(sizeof(_Complex float)*L*L);
  return ptr;
}

/*
 * allocates a new folded field and returns its starting address
 */
void *
new_folded_field(int L)
{
  void *ptr = alloc(sizeof(float)*L*L*2);
  return ptr;
}

/*
 * allocates a new U(1) gauge field and returns its starting address
 */
void *
new_links(int L)
{
  void *ptr = alloc(sizeof(_Complex float)*L*L*ND);
  return ptr;
}

/*
 * allocates a new folded U(1) gauge field and returns its starting address
 */
void *
new_folded_links(int L)
{
  void *ptr = alloc(sizeof(float)*L*L*ND*2);
  return ptr;
}

/*
 * Fills u with random entries on the unit circle, gaussianly
 * distributed around 1 + 0*i, using Box-Mueller
 */
void
rand_links(int L, _Complex float *u)
{
  for(int i=0; i<L*L; i++) 
    for(int d=0; d<ND; d++) {
      int id = d + i*ND;
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
rand_field(int L, _Complex float *x)
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
zero_field(int L, _Complex float *x)
{
  for(int i=0; i<L*L; i++) {
    x[i] = 0.0;
  }
  return;
}

/*
 * Folds field x, of length L*L, and returns in y
 */
void
fold_field(int L, float *y, _Complex float *x)
{
  for(int iy=0; iy<L/4; iy++)
    for(int ix=0; ix<L; ix++)
      for(int reim=0; reim<2; reim++)
	for(int jy=0; jy<4; jy++) {
	  int v0 = (iy+jy*L/4)*L + ix;
	  int v1 = iy*L*8 + ix*8 + reim*4 + jy;
	  y[v1] = reim == 0 ? creal(x[v0]) : cimag(x[v0]);
	}
  return;
}

/*
 * Folds gauge field u, of length L*L, and returns in v
 */
void
fold_links(int L, float *v, _Complex float *u)
{
  for(int iy=0; iy<L/4; iy++)
    for(int ix=0; ix<L; ix++)
      for(int d=0; d<ND; d++)
	for(int reim=0; reim<2; reim++)
	  for(int jy=0; jy<4; jy++) {
	    int v0 = (iy+jy*L/4)*L + ix;
	    int v1 = iy*L*8*ND + ix*8*ND + d*8 + reim*4 + jy;
	    v[v1] = reim == 0 ? creal(u[v0*ND+d]) : cimag(u[v0*ND+d]);
	  }
  return;
}

/*
 * Applies U(1) gauge laplacian to phi_in, with background field u,
 * and returns in phi_out
 */
void
lapl(int L, _Complex float *phi_out, _Complex float *phi_in, _Complex float *u)
{
#pragma omp parallel for
  for(int y=0; y<L; y++)
    for(int x=0; x<L; x++) {
      int v00 = y*L + x;
      int v0p = y*L + (x+1)%L;
      int vp0 = ((y+1)%L)*L + x;
      int v0m = y*L + (L+x-1)%L;
      int vm0 = ((L+y-1)%L)*L + x;
      
      _Complex float p = 0;
      p  = phi_in[v0p]*u[UIDX(v00, 0)];
      p += phi_in[vp0]*u[UIDX(v00, 1)];
      p += phi_in[v0m]*conj(u[UIDX(v0m, 0)]);
      p += phi_in[vm0]*conj(u[UIDX(vm0, 1)]);
      phi_out[v00] = 4*phi_in[v00] - p;
    }
      
  return;
}

/*
 * Applies U(1) gauge laplacian to folded field phi_in, with
 * background folded gauge field u, and returns in folded field
 * phi_out
 */
void
lapl_folded(int L, float *phi_out, float *phi_in, float *u)
{
#pragma omp parallel for
  for(int y=0; y<L/4; y++)
    for(int x=0; x<L; x++) {
      int v00 = y*L + x;
      int v0p = y*L + (x+1)%L;
      int v0m = y*L + (L+x-1)%L;
      int vp0 = ((L/4+y+1)%(L/4))*L + x;
      int vm0 = ((L/4+y-1)%(L/4))*L + x;

      float *q00r = &phi_out[v00*8];
      float *q00i = &phi_out[v00*8+4];

      float *p00r = &phi_in[v00*8];
      float *p00i = &phi_in[v00*8+4];

      float p0pr[4], p0pi[4];
      float p0mr[4], p0mi[4];
      float pp0r[4], pp0i[4];
      float pm0r[4], pm0i[4];
      
      for(int iy=0; iy<4; iy++) {
	p0pr[iy] = phi_in[v0p*8+iy];
	p0pi[iy] = phi_in[v0p*8+4+iy];

	pp0r[iy] = phi_in[vp0*8+iy];
	pp0i[iy] = phi_in[vp0*8+4+iy];

	p0mr[iy] = phi_in[v0m*8+iy];
	p0mi[iy] = phi_in[v0m*8+4+iy];

	pm0r[iy] = phi_in[vm0*8+iy];
	pm0i[iy] = phi_in[vm0*8+4+iy];
      }
      
      /* Links in direction \hat{0} */
      float *u000r = &u[v00*8*ND];
      float *u000i = &u[v00*8*ND+4];
      float u00mr[4], u00mi[4];
      
      for(int iy=0; iy<4; iy++) {
	u00mr[iy] = u[v0m*8*ND+iy];
	u00mi[iy] = u[v0m*8*ND+4+iy];
      }
      /* Links in direction \hat{1} */
      float *u100r = &u[v00*8*ND+8];
      float *u100i = &u[v00*8*ND+8+4];

      float u1m0r[4], u1m0i[4];
      for(int iy=0; iy<4; iy++) {
	u1m0r[iy] = u[vm0*8*ND+8+iy];
	u1m0i[iy] = u[vm0*8*ND+8+4+iy];
      }
      
      if(y == 0)
	for(int iy=3; iy>0; iy--) {
	  float swap;

	  swap = pm0r[iy];
	  pm0r[iy] = pm0r[iy-1];
	  pm0r[iy-1] = swap;
	  
	  swap = pm0i[iy];
	  pm0i[iy] = pm0i[iy-1];
	  pm0i[iy-1] = swap;

	  swap = u1m0r[iy];
	  u1m0r[iy] = u1m0r[iy-1];
	  u1m0r[iy-1] = swap;
	  
	  swap = u1m0i[iy];
	  u1m0i[iy] = u1m0i[iy-1];
	  u1m0i[iy-1] = swap;
	}
      
      if(y == L/4-1)
	for(int iy=0; iy<3; iy++) {
	  float swap;

	  swap = pp0r[iy];
	  pp0r[iy] = pp0r[iy+1];
	  pp0r[iy+1] = swap;
	  
	  swap = pp0i[iy];
	  pp0i[iy] = pp0i[iy+1];
	  pp0i[iy+1] = swap;	  
	}
	
      for(int iy=0; iy<4; iy++) {
	q00r[iy] = 4*p00r[iy];
	q00r[iy] -= p0pr[iy]*u000r[iy] - p0pi[iy]*u000i[iy];
	q00r[iy] -= pp0r[iy]*u100r[iy] - pp0i[iy]*u100i[iy];
	q00r[iy] -= p0mr[iy]*u00mr[iy] + p0mi[iy]*u00mi[iy];
	q00r[iy] -= pm0r[iy]*u1m0r[iy] + pm0i[iy]*u1m0i[iy];

	q00i[iy] = 4*p00i[iy];
	q00i[iy] -= p0pi[iy]*u000r[iy] + p0pr[iy]*u000i[iy];
	q00i[iy] -= pp0i[iy]*u100r[iy] + pp0r[iy]*u100i[iy];
	q00i[iy] -= p0mi[iy]*u00mr[iy] - p0mr[iy]*u00mi[iy];
	q00i[iy] -= pm0i[iy]*u1m0r[iy] - pm0r[iy]*u1m0i[iy];
      }
    }
  return;
}

/*
 * returns y = x^H x for vector x of length L*L
 */
float
xdotx(int L, _Complex float *x)
{
  float y = 0; 
  for(int i=0; i<L*L; i++) {
    y += creal(x[i])*creal(x[i]) + cimag(x[i])*cimag(x[i]);
  }
  return y;
}

/*
 * returns y = x^H x for folded vector x of length L*L
 */
float
xdotx_folded(int L, float *x)
{
  float y = 0; 
  for(int iv=0; iv<L*L/4; iv++) {
    float *xr = &x[iv*8];
    float *xi = &x[iv*8+4];
    for(int iy=0; iy<4; iy++) {
      y += xr[iy]*xr[iy] + xi[iy]*xi[iy];
    }
  }
  return y;
}

/*
 * returns z = x^H y for vectors x, y of length L*L
 */
_Complex float
xdoty(int L, _Complex float *x, _Complex float *y)
{
  float z = 0; 
  for(int i=0; i<L*L; i++)
    z += conj(x[i])*y[i];
  
  return z;
}

/*
 * returns z = x^H y for folded vectors x, y of length L*L
 */
_Complex float
xdoty_folded(int L, float *x, float *y)
{
  float zr = 0; 
  float zi = 0; 
  for(int iv=0; iv<L*L/4; iv++) {
    float *yr = &y[iv*8];
    float *yi = &y[iv*8+4];
    float *xr = &x[iv*8];
    float *xi = &x[iv*8+4];
    for(int iy=0; iy<4; iy++) {
      zr += xr[iy]*yr[iy] + xi[iy]*yi[iy];
      zi += xr[iy]*yi[iy] - xi[iy]*yr[iy];
    }
  }
  return zr + _Complex_I*zi;
}

/*
 * returns y = x - y for vectors y, x, of length L*L
 */
void
xmy(int L, _Complex float *x, _Complex float *y)
{
  for(int i=0; i<L*L; i++) {
    y[i] = x[i] - y[i];
  }
  return;
}

/*
 * returns y = x - y for folded vectors y, x, of length L*L
 */
void
xmy_folded(int L, float *x, float *y)
{
  for(int i=0; i<L*L*2; i++) {
    y[i] = x[i] - y[i];
  }
  return;
}

/*
 * returns y = x for vectors y, x, of length L*L
 */
void
xeqy(int L, _Complex float *x, _Complex float *y)
{
  for(int i=0; i<L*L; i++)
    x[i] = y[i];
  
  return;
}

/*
 * returns y = x for folded vectors y, x, of length L*L
 */
void
xeqy_folded(int L, float *x, float *y)
{
  for(int i=0; i<L*L*2; i++)
    x[i] = y[i];
  
  return;
}

/*
 * returns y = a*x+y for vectors y, x, of length L*L and scalar a
 */
void
axpy(int L, _Complex float a, _Complex float *x, _Complex float *y)
{
  for(int i=0; i<L*L; i++)
    y[i] = a*x[i] + y[i];
  
  return;
}

/*
 * returns y = a*x+y for folded vectors y, x, of length L*L and scalar a
 */
void
axpy_folded(int L, _Complex float a, float *x, float *y)
{
  for(int iv=0; iv<L*L/4; iv++) {
    float *yr = &y[iv*8];
    float *yi = &y[iv*8+4];
    float *xr = &x[iv*8];
    float *xi = &x[iv*8+4];
    float ar = creal(a);
    float ai = cimag(a);
    for(int iy=0; iy<4; iy++) {
      yr[iy] =  ar*xr[iy] - ai*xi[iy] + yr[iy];
      yi[iy] =  ai*xr[iy] + ar*xi[iy] + yi[iy];      
    }
  }
  return;
}

/*
 * returns y = x+a*y for vectors y, x, of length L*L and scalar a
 */
void
xpay(int L, _Complex float *x, _Complex float a, _Complex float *y)
{
  for(int i=0; i<L*L; i++)
    y[i] = x[i] + a*y[i];
  
  return;
}

/*
 * returns y = x+a*y for folded vectors y, x, of length L*L and scalar a
 */
void
xpay_folded(int L, float *x, _Complex float a, float *y)
{
  for(int iv=0; iv<L*L/4; iv++) {
    float *yr = &y[iv*8];
    float *yi = &y[iv*8+4];
    float *xr = &x[iv*8];
    float *xi = &x[iv*8+4];
    float ar = creal(a);
    float ai = cimag(a);
    for(int iy=0; iy<4; iy++) {
      float yyr = yr[iy];
      float yyi = yi[iy];
      yr[iy] = xr[iy] + ar*yyr - ai*yyi;
      yi[iy] = xi[iy] + ai*yyr + ar*yyi;      
    }
  }
  return;
}


/*
 * Solves lapl(u) x = b, for x, given b, using Conjugate Gradient
 */
void
cg(int L, _Complex float *x, _Complex float *b, _Complex float *u)
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
  printf(" Converged after %6d iterations, res = %+e\n", iter, rr/bb);  
  printf(" Time in lapl(): %+6.3e sec/iter\n", t_lapl/(double)iter);  

  free(r);
  free(p);
  free(Ap);
  return;
}

/*
 * Solves lapl(u) x = b, for x, given b, using Conjugate Gradient, for folded fields
 */
void
cg_folded(int L, float *x, float *b, float *u)
{
  int max_iter = 100;
  float tol = 1e-6;

  /* Temporary fields needed for CG */
  float *r = new_folded_field(L);
  float *p = new_folded_field(L);
  float *Ap = new_folded_field(L);

  /* Initial residual and p-vector */
  lapl_folded(L, r, x, u);
  xmy_folded(L, b, r);
  xeqy_folded(L, p, r);

  /* Initial r-norm and b-norm */
  float rr = xdotx_folded(L, r);  
  float bb = xdotx_folded(L, b);
  double t_lapl = 0;
  int iter = 0;
  for(iter=0; iter<max_iter; iter++) {
    printf(" %6d, res = %+e\n", iter, rr/bb);
    if(sqrt(rr/bb) < tol)
      break;
    
    double t = stop_watch(0);
    lapl_folded(L, Ap, p, u);
    t_lapl += stop_watch(t);

    float pAp = xdoty_folded(L, p, Ap);
    float alpha = rr/pAp;
    axpy_folded(L, alpha, p, x);
    axpy_folded(L, -alpha, Ap, r);
    float r1r1 = xdotx_folded(L, r);
    float beta = r1r1/rr;
    xpay_folded(L, r, beta, p);
    rr = r1r1;
  }

  /* Recompute residual after convergence */
  lapl_folded(L, r, x, u);
  xmy_folded(L, b, r);
  rr = xdotx_folded(L, r);

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
  int L = (int)strtoul(argv[1], &e, 10);
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

  float *b_folded = new_folded_field(L); 
  float *x_folded = new_folded_field(L); 
  float *u_folded = new_folded_links(L); 

  fold_field(L, x_folded, x);
  fold_field(L, b_folded, b);
  fold_links(L, u_folded, u);
  
  cg_folded(L, x_folded, b_folded, u_folded);
  
  free(b);
  free(x);
  free(u);

  free(b_folded);
  free(x_folded);
  free(u_folded);
  return 0;
}
