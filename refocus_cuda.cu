
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <complex>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <cufft.h>
#include <fstream>
#include <vector>
#include <ctime>
#include <numeric> 
#include <math.h>
#include "func_file.h"

#define PI 3.14159265359

using namespace std;

void fft_propogate(cufftComplex* phimap_in, cufftComplex* phimap_out, double &d, float &nm, float &res, int sizex, int sizey)
{
	int blocksInX = (sizex + 32 - 1) / 32;
	int blocksInY = (sizey + 32 - 1) / 32;
	dim3 grid(blocksInX, blocksInY);
	dim3 block(32, 32);
	cufftComplex *d_p_in , *d_p_out;
	cudaMalloc(&d_p_in, sizeof(cufftComplex)*sizex*sizey);
	cudaMalloc(&d_p_out, sizeof(cufftComplex)*sizex*sizey);
	cudaMemcpy(d_p_in, phimap_in, sizeof(cufftComplex)*sizex*sizey, cudaMemcpyHostToDevice);
	
	fft_propogate_cu << <grid, block >> > (d_p_in, d_p_out, d, nm, res, sizex, sizey);

	cudaThreadSynchronize();
	cufftHandle plan;
	cufftPlan2d(&plan,sizex , sizey, CUFFT_C2C);
	cufftExecC2C(plan, (cufftComplex *)d_p_out, (cufftComplex *)d_p_out, CUFFT_INVERSE);  // must divide number of elements(sizex*sizey) to return correct value 

	cudaMemcpy(phimap_out, d_p_out, sizeof(cufftComplex)*sizex*sizey, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 256; i++)
	//{
	//	for (int j = 0; j < 256; j++)
	//	{
	//		cout<<" (" << phimap_out[i*256+j].x << ", " << phimap_out[i * 256 + j].y << ") ";
	//	}
	//	cout << "\n";
	//}
	cufftDestroy(plan);
	cudaFree(d_p_in);
	cudaFree(d_p_out);
}


float tuma_calculation(cufftComplex* fulltuma, int sizex, int sizey)
{
	float  *d_tuma_coeff , *d_tuma_mean;
	float tuma_coeff , tuma_mean;
	int blocksInX = (sizex*sizey + 32 - 1) / 32;
	dim3 grid(blocksInX,1,1);
	dim3 block(32,1,1);

	//cout << fulltuma[5].x;
	cudaMalloc((void**)&d_tuma_coeff, sizeof(float));
	cudaMalloc((void**)&d_tuma_mean, sizeof(float));
	cufftComplex *d_fulltuma ;
	cudaMalloc(&d_fulltuma, sizeof(cufftComplex)*sizex*sizey);
	//cudaMemcpy(d_std_mean, std_mean, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_fulltuma, fulltuma, sizeof(cufftComplex)*sizex*sizey, cudaMemcpyHostToDevice);
	tuma_calculation_cu<<<grid , block >>>(d_fulltuma, sizex, sizey, d_tuma_coeff,d_tuma_mean);
	//tuma_calculation_cu << <1,1 >> > (d_fulltuma, sizex, sizey, d_tuma_coeff);
	//cudaThreadSynchronize();

	cudaMemcpy(&tuma_coeff, d_tuma_coeff, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&tuma_mean, d_tuma_mean, sizeof(float), cudaMemcpyDeviceToHost);

	tuma_coeff -= tuma_mean * tuma_mean;
	//cout << tuma_coeff <<"\n";

	cudaFree(d_fulltuma);
	cudaFree(d_tuma_coeff);
	cudaFree(d_tuma_mean);
	return tuma_coeff/tuma_mean;
}

float tuma_calculation_cpp(cufftComplex* fulltuma, int sizex, int sizey)
{
	// calculate average
	float sum = 0, std = 0;

	//float average = accumulate(fulltuma.begin(), fulltuma.end(), 0.0)/(sizex*sizey); //accumulate(start,end,initial_num)
	for (int i = 0; i < sizex*sizey; i++)
	{
		sum += fulltuma[i].x;
		std += fulltuma[i].x*fulltuma[i].x;
	}
	float average = sum / (sizex*sizey);
	std /= (sizex*sizey);
	std -= average * average;
	//std /= (sizex*sizey);
	return std / average;
}


void main(cufftComplex* phimap_in, int sizex, int sizey, double start_pos, double end_pos)
{
	long starttime = clock();
	sizex = 256;
	sizey = 256;
	// because 256*256 is to big to use static memory therefore have to allocate an dynamic memory first 
	phimap_in = new cufftComplex[sizex*sizey];
	cufftComplex* phimap_out = new cufftComplex[sizex*sizey];
	//vector<double> phimap(sizex*sizey);

	double readinreal;
	double readinimag;
	ifstream infile;
	ifstream infile2;
	//infile.open("real.txt");
	infile.open("D:\\cuda_learn\\refocus_cuda_ver\\real.txt");
	//infile2.open("imag.txt");
	infile2.open("D:\\cuda_learn\\refocus_cuda_ver\\imag.txt");
	for (int i = 0; i < sizex*sizey; i++)
	{
		infile >> phimap_in[i].x;
		infile2 >> phimap_in[i].y;
		//cout << phimap_in[i].x << " ";
	}
	infile.close();
	infile2.close();
	//cout << "read ok";

	// parameter
	float nm = 1.333;
	float res = 0.532;
	start_pos = -20;
	end_pos = 20;
	float step = 0.1;
	int output_len = (end_pos - start_pos) / step;

	
	vector <float> tuma;
	vector <double> Position;

	//fft_propogate(phimap_in, phimap_out, start_pos, nm, res, sizex, sizey);
	//float tuma_coeff = tuma_calculation(phimap_out, sizex, sizey);
	//float tuma_coeff = tuma_calculation_cpp(phimap_out, sizex, sizey);
	//cout <<"@@" << tuma_coeff << "@@";

	
	//cout << "start cal";
	for (start_pos; start_pos < end_pos; start_pos += step)
	{
		fft_propogate(phimap_in, phimap_out, start_pos, nm, res, sizex, sizey);
		//float tuma_coeff = tuma_calculation(phimap_out, sizex, sizey);
		float tuma_coeff = tuma_calculation_cpp(phimap_out, sizex, sizey);
		tuma.push_back(tuma_coeff);
		Position.push_back(start_pos);
		//cout <<start_pos << " \n";
		tuma.size();
	}
	
	//cout << "cal max";
	// find max num in tuna and return position
	vector<float>::iterator biggest = max_element(begin(tuma), end(tuma));
	//cout << "Max element is " << *biggest << " at position " << distance(std::begin(tuma), biggest) << endl;

	// return final position
	double return_pos = Position[distance(std::begin(tuma), biggest)];
	cout << "best position : " << return_pos;
	fft_propogate(phimap_in, phimap_in, return_pos, nm, res, sizex, sizey);
	long finishtime = clock();
	cout << "@@@@@  " << (finishtime - starttime) << "  @@@@@";
	delete[] phimap_in;
	delete[] phimap_out;
}

