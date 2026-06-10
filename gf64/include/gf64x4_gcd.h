#ifndef GF64X4_GCD_H
#define GF64X4_GCD_H
#include "gf64x4.h"

/* 4-Way coprime test using Berstein-Yang's GCD Algorithm
*  We assume that all the batched polynomials has the same degree
*  While this is true in our protocol, if not, then the algorithm does not work as expected
*/

void gf64x4_coprime_test(const __m512i *a, int na, const __m512i *b, int nb,
                              __m512i *work, int deg_gt0[4]);

#endif