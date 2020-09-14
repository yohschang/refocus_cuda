
#include <complex>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <ctime>
#include <numeric> 
#include <math.h>

#define PI acos(-1)
#define I complex<double>(0,1)

using namespace std;

int Powerof2(int n, int *m, int *twopm)
{
	if (n <= 1)
	{
		*m = 0;
		*twopm = 1;
		return(false);
	}

	*m = 1;
	*twopm = 2;
	do {
		(*m)++;
		(*twopm) *= 2;
	} while (2 * (*twopm) <= n);
	//cout << *m;

	if (*twopm != n)
		return(false);
	else
		return(true);
}
int FFT(int dir, int m, double *x, double *y)
{
	long nn, i, i1, j, k, i2, l, l1, l2;
	double c1, c2, tx, ty, t1, t2, u1, u2, z;

	//Calculate the number of points 
	nn = 1;
	for (i = 0; i < m; i++)
		nn *= 2;
	// Do the bit reversal 
	i2 = nn >> 1;
	j = 0;
	for (i = 0; i < nn - 1; i++)
	{
		if (i < j)
		{
			tx = x[i];
			ty = y[i];
			x[i] = x[j];
			y[i] = y[j];
			x[j] = tx;
			y[j] = ty;
		}
		k = i2;
		while (k <= j)
		{
			j -= k;
			k >>= 1;
		}
		j += k;
	}

	//Compute the FFT 
	c1 = -1.0;
	c2 = 0.0;
	l2 = 1;
	for (l = 0; l < m; l++)
	{
		l1 = l2;
		l2 <<= 1;
		u1 = 1.0;
		u2 = 0.0;
		for (j = 0; j < l1; j++)
		{
			for (i = j; i < nn; i += l2)
			{
				i1 = i + l1;
				t1 = u1 * x[i1] - u2 * y[i1];
				t2 = u1 * y[i1] + u2 * x[i1];
				x[i1] = x[i] - t1;
				y[i1] = y[i] - t2;
				x[i] += t1;
				y[i] += t2;
			}
			z = u1 * c1 - u2 * c2;
			u2 = u1 * c2 + u2 * c1;
			u1 = z;
		}
		c2 = sqrt((1.0 - c1) / 2.0);
		if (dir == 1)
			c2 = -c2;
		c1 = sqrt((1.0 + c1) / 2.0);
	}

	// Scaling for forward transform 
	if (dir == 1)
	{
		for (i = 0; i < nn; i++)
		{
			x[i] /= (double)nn;
			y[i] /= (double)nn;

		}
	}
	return(true);
}
int FFT2D(vector<complex<double> > &c, int nx, int ny, int dir)
{
	int m, twopm;
	double *realC, *imagC;

	// Transform the rows 
	realC = (double *)malloc(nx * sizeof(double));
	imagC = (double *)malloc(nx * sizeof(double));
	if (realC == NULL || imagC == NULL)
		return(false);
	if (!Powerof2(nx, &m, &twopm) || twopm != nx)
		return(false);

	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			realC[i] = (double)real(c[i*ny + j]);
			imagC[i] = (double)imag(c[i*ny + j]);
		}

		FFT(dir, m, realC, imagC);

		for (int i = 0; i < nx; i++)
		{
			c[i*ny + j] = complex<float>((float)realC[i], (float)imagC[i]);
		}
	}

	// Transform the columns
	realC = (double *)realloc(realC, nx * sizeof(double));
	imagC = (double *)realloc(imagC, nx * sizeof(double));
	if (realC == NULL || imagC == NULL)
		return(false);
	if (!Powerof2(nx, &m, &twopm) || twopm != nx)
		return(false);
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			realC[j] = (double)real(c[i*ny + j]);
			imagC[j] = (double)imag(c[i*ny + j]);
		}

		FFT(dir, m, realC, imagC);

		for (int j = 0; j < ny; j++)
		{
			c[i*ny + j] = complex<float>((float)realC[j], (float)imagC[j]);
			//cout << c[i*ny + j] << " ";
		}
		//cout << "\n";
	}
	free(realC);
	free(imagC);

	return(true);
}

void fft_propogate(vector<complex<double> > &phimap, double &d, float &nm, float &res, int sizex, int sizey)
{
	double km = (2 * PI*nm) / res;
	double kx_o = -1, ky_o = -1, kx, ky;
	complex<double> fstemp;
	for (int i = 0; i < sizex; i++)
		//for (int i = 0; i < 1; i++)
	{
		//calculate fftfreqx
		if (sizex % 2 != 0)
		{
			if (i < (sizex + 1) / 2) kx_o++;
			else if (i > (sizex + 1) / 2) kx_o++;
			else kx_o *= -1;
		}
		else if (sizex % 2 == 0)
		{
			if (i < (sizex / 2)) kx_o++;
			else if (i > (sizex / 2)) kx_o++;
			else { kx_o++; kx_o *= -1; }
		}
		for (int j = 0; j < sizey; j++)
			//for (int j = 0; j < 1; j++)
		{
			//calculate fftfreqy
			if (sizey % 2 != 0)
			{
				if (j < (sizey + 1) / 2) ky_o++;
				else if (j > (sizey + 1) / 2) ky_o++;
				else ky_o *= -1;
			}
			else if (sizey % 2 == 0)
			{
				if (j < (sizey / 2)) ky_o++;
				else if (j > (sizey / 2)) ky_o++;
				else { ky_o++; ky_o *= -1; }
			}

			kx = (kx_o / sizex) * 2 * PI;
			ky = (ky_o / sizey) * 2 * PI;
			double root_km = km * km - kx * kx - ky * ky;
			bool rt0 = root_km > 0;

			if (root_km > 0)
			{
				fstemp = exp(I*(sqrt(root_km * rt0) - km)*d);
			}
			else
			{
				fstemp = 0;
			}
			phimap[i*sizex + j] *= fstemp;
			//cout << phimap[i*sizex + j] << " ";
		}
	}
	;
	FFT2D(phimap, sizex, sizey, 1);  //-1 fft //1 ifft
}


double tuma_calculation(vector<complex<double> > &fulltuma, int sizex, int sizey)
{
	// calculate average
	float sum = 0, std = 0;

	//float average = accumulate(fulltuma.begin(), fulltuma.end(), 0.0)/(sizex*sizey); //accumulate(start,end,initial_num)
	for (int i = 0; i < sizex*sizey; i++)
	{
		sum += fulltuma[i].real();
	}
	float average = sum / (sizex*sizey);
	for (int i = 0; i < sizex*sizey; i++)
	{
		std += ((fulltuma[i].real() - average)*(fulltuma[i].real() - average));
	}
	std /= (sizex*sizey);
	//cout << average << " " << std;
	return std / average;
}

void main(float return_pos, int sizex, int sizey, double start_pos, double end_pos)
{
	sizex = 256;
	sizey = 256;
	// because 256*256 is to big to use static memory therefore have to allocate an dynamic memory first 
	vector<complex<double> > phimap(sizex*sizey);
	vector<complex<double> > phimapcopy(sizex*sizey);
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
		infile >> readinreal;
		//infile >> phimap[i];
		infile2 >> readinimag;

		phimap[i].real(readinreal);
		phimap[i].imag(readinimag);
		//cout << phimap[i] << " ";
	}
	infile.close();
	infile2.close();
	cout << "read ok";

	// parameter
	float nm = 1.333;
	float res = 0.532;
	start_pos = -200;
	end_pos = 200;

	vector <double> tuma;
	vector <float> Position;
	//double tuma_coeff = tuma_calculation(phimap, 4,4);
	cout << "start cal";
	long starttime = clock();
	for (start_pos; start_pos < end_pos; start_pos += 1)
	{
		phimapcopy.assign(phimap.begin(), phimap.end());
		fft_propogate(phimapcopy, start_pos, nm, res, sizex, sizey);
		double tuma_coeff = tuma_calculation(phimapcopy, sizex, sizey);
		tuma.push_back(tuma_coeff);
		Position.push_back(start_pos);
		cout << start_pos << " " <<tuma_coeff << " //";
		//tuma.size();
	}
	long finishtime = clock();
	cout << "cal max";
	// find max num in tuna and return position
	vector<double>::iterator biggest = max_element(begin(tuma), end(tuma));
	//cout << "Max element is " << *biggest << " at position " << distance(std::begin(tuma), biggest) << endl;

	// return final position
	return_pos = Position[distance(std::begin(tuma), biggest)];
	cout << "best position : " << return_pos;
	cout << "@@@@@  " << (finishtime - starttime) << "  @@@@@";
}

