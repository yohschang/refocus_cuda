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

#define PI 3.14159265359

using namespace std;

__device__ cufftComplex com_exp(cufftComplex z)
{
	cufftComplex res;
	float t = expf(z.x);
	float z_cos = cosf(z.y);
	float z_sin = sinf(z.y);
	res.x = z_cos * t;
	res.y = z_sin * t;
	return res;
}

__device__ cufftComplex com_mul(cufftComplex a, cufftComplex b)
{
	cufftComplex result;
	result.x = (a.x*b.x) - (a.y*b.y);
	result.y = (a.x*b.y) + (a.y*b.x);
	return result;
}

__global__ void fft_propogate_cu(cufftComplex*p_in, cufftComplex*p_out, double d, float nm, float res, int sizex, int sizey)
{
	
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	double km = (2 * PI*nm) / res;
	double kx_o, ky_o, kx, ky;
	cufftComplex I;

	
	if (i < sizex && j < sizey)
	{
		if (sizex % 2 != 0)
		{
			if (i < (sizex + 1) / 2)
				kx_o = i;
			else if (i >= (sizex + 1) / 2)
				kx_o = -(sizex - i);
		}
		else if (sizex % 2 == 0)
		{
			if (i < (sizex / 2))
				kx_o = i;
			else if (i >= (sizex / 2))
				kx_o = -(sizex - i);
		}

		if (sizey % 2 != 0)
		{
			if (j < (sizey + 1) / 2)
				ky_o = j;
			else if (j >= (sizey + 1) / 2)
				ky_o = -(sizey - j);
		}
		else if (sizey % 2 == 0)
		{
			if (j < (sizey / 2))
				ky_o = j;
			else if (j >= (sizey / 2))
				ky_o = -(sizey - j);
		}

		kx = (kx_o / sizex) * 2 * PI;
		ky = (ky_o / sizey) * 2 * PI;
		double root_km = km * km - kx * kx - ky * ky;
		bool rt0 = root_km > 0;

		if (root_km > 0)
		{
			I.x = 0;
			I.y = (sqrt(root_km * rt0) - km)*d;
			p_out[i*sizex + j] = com_mul(p_in[i*sizex + j], com_exp(I));
		}
		else
		{
			p_out[i*sizex + j].x = 0.0;
			p_out[i*sizex + j].y = 0.0;
		}
		//p_out[i*sizex + j].x = kx_o;
		//p_out[i*sizex + j].y = ky_o;
	}

}

__global__ void tuma_calculation_cu(cufftComplex* fulltuma, int sizex, int sizey, float *tuma_coeff , float *tuma_mean)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	float mean, std  ;
	int x, y;
	x = sizex;
	y = sizey;
	if (idx < sizex*sizey)
	{	
		atomicAdd(tuma_mean, fulltuma[idx].x / (x*y));
		atomicAdd(tuma_coeff, (fulltuma[idx].x * fulltuma[idx].x) / (x*y));
	}
	__syncthreads();
	//*tuma_coeff -= (*tuma_mean * *tuma_mean);

	//if (idx < sizex*sizey)
	//{
	//	atomicAdd(tuma_coeff, ((fulltuma[idx].x - *tuma_mean)*(fulltuma[idx].x - *tuma_mean)));
	//}
	//*tuma_coeff /= (x*y );

}

