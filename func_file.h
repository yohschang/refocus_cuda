#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <complex>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <cufft.h>
#include <fstream>
#include <vector>
#include <numeric> 
#include <math.h>

__global__ void tuma_calculation_cu(cufftComplex* fulltuma, int sizex, int sizey, float *tuma_coeff, float *tuma_mean);
__global__ void fft_propogate_cu(cufftComplex*p_in, cufftComplex*p_out, double d, float nm, float res, int sizex, int sizey); __device__ cufftComplex com_exp(cufftComplex z);
__device__ cufftComplex com_mul(cufftComplex a, cufftComplex b);