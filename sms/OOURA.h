#ifndef _OOURA_H
#define _OOURA_H

#ifdef DOUBLE_PRECISION
#define sfloat double
#else
#define sfloat float
#endif

#define NMAX 8192
#define NMAXSQRT 64

void rdft(int n, int isgn, sfloat *a, int *ip, sfloat *w);

void makewt(int nw, int *ip, sfloat *w);
void makect(int nc, int *ip, sfloat *c);
void bitrv2(int n, int *ip, sfloat *a);
void cftfsub(int n, sfloat *a, sfloat *w);
void cftbsub(int n, sfloat *a, sfloat *w);
void rftfsub(int n, sfloat *a, int nc, sfloat *c);
void rftbsub(int n, sfloat *a, int nc, sfloat *c);

void cft1st(int n, sfloat *a, sfloat *w);
void cftmdl(int n, int l, sfloat *a, sfloat *w);

#endif /* _OURA_H */
