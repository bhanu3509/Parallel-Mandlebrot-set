#include"Mandlebrot.h"
#include<iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include<cstdint>
#include<complex>
#include <cstdio>
 using namespace std;

  struct Complex {
	 long double real;
	 long double imaginary;
 };
 Complex com;
struct Pixel
{
	unsigned char red;
	unsigned char green;
	unsigned char blue;
};

static const int MAX_PIXEL_VAL = 255;


static const int Image_Width = 5000;
static const int Image_Height = 5000;
static const int Max_Iterations = 1000;

static const Complex Focus_Point = {com.real = -0.5, com.imaginary = 0 };
static const long double Zoom = 2;

// We use the coloring schema outlined from https://solarianprogrammer.com/2013/02/28/mandelbrot-set-cpp-11/
double calc_colors(Pixel* colors) {
	for (int i = 0; i < Max_Iterations; i++) {
		double t = (double)i / Max_Iterations;

		colors[i][0] = (unsigned char)(9 * (1 - t) * t * t * t * MAX_PIXEL_VAL);
		colors[i][1] = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * MAX_PIXEL_VAL);
		colors[i][2] = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * MAX_PIXEL_VAL);
	}
};

/* The Mandelbrot set is defined by all numbers which do not diverge for fc(z) = z^2 + c,
 * where C is a complex number. Generally, we run the algorithm until we hit a cutoff number of iterations.
 * We can end the iterations early if we know that the sum of the complex coefficients is <= 4.
 * because if that happens we know it'll diverge.
 *
 * To draw the set, we map the real value to the x-axis, and the imaginary value to the y-axis.
 * We then use the number of iterations to escape to calculate the color of the pixel
 *
 * To convert the resulting PPM, you may use http://www.imagemagick.org
 */

int main(int argc, const char** argv) {
	Pixel** pixels;
	pixels = static_cast<Pixel **>(malloc(Image_Width * Image_Height * sizeof(Pixel)));

	Pixel colors[Max_Iterations + 1][Max_Iterations];
	calc_colors(reinterpret_cast<Pixel *>(colors));

    // Calculate scaling values to map the bounds of the Mandelbrot area to the pixel grid
	const Complex min_bounds = { com.real = Focus_Point.real - Zoom, com.imaginary = Focus_Point.imaginary - Zoom };
	const Complex max_bounds = { com.real = Focus_Point.real + Zoom, com.imaginary = Focus_Point.imaginary + Zoom };
	const Complex scale = {
			com.real = (max_bounds.real - min_bounds.real) / Image_Width,
			com.imaginary = (max_bounds.real - min_bounds.real) / Image_Height
	};
	int num_thread = 4;
#ifdef _LODE_BALANCE_ANALYSIS_
	double timer[num_thread];
#else
	double s = omp_get_wtime();
#endif

#pragma omp parallel for schedule(dynamic, 10) private(c) collapse(2)
	// Loop through the image pixels
	for (int img_y = 0; img_y < Image_Height; img_y++) {
		for (int img_x = 0; img_x < Image_Width; img_x++) {
			// Find the value of C in the Mandelbrot range corresponding to this pixel
#ifdef _LODE_BALANCE_ANALYSIS_
			double s = omp_get_wtime();
#endif
			Complex c = {
					c.real = min_bounds.real + img_x * scale.real,
					c.imaginary = min_bounds.imaginary + img_y * scale.imaginary
			};

			// Check if the current pixel is in the Mandelbrot set
			// We use the optimizations from https://randomascii.wordpress.com/2011/08/13/faster-fractals-through-algebra/
			Complex z = { z.real = 0, z.imaginary = 0 };
			Complex z_squared = { z_squared.real = 0, z_squared.imaginary = 0 };

			int iterations = 0;
			while (z_squared.real + z_squared.imaginary <= 4 && iterations < Max_Iterations) {
				z.imaginary = z.real * z.imaginary;
				z.imaginary += z.imaginary;
				z.imaginary += c.imaginary;

				z.real = z_squared.real - z_squared.imaginary + c.real;

				z_squared.real = z.real * z.real;
				z_squared.imaginary = z.imaginary * z.imaginary;

				iterations++;
#ifdef _LODE_BALANCE_ANALYSIS_
				timer[omp_get_thread_num()] += omp_get_wtime() - s;
#endif
#pragma omp critical
			}

			pixels[img_y * Image_Width + img_x][0] = colors[iterations][0];
			pixels[img_y * Image_Width + img_x][1] = colors[iterations][1];
			pixels[img_y * Image_Width + img_x][2] = colors[iterations][2];
		}
	}
#ifdef _LODE_BALANCE_ANALYSIS_
	for (int i = 0; i < num_thread; ++i)
		printf("#%d : %lf\n", i, timer[i]);
	// cout << fixed << "#" << i << ": " << timer[i] << endl;
#else
	printf("%lf\n", omp_get_wtime() - s);
	// cout << omp_get_wtime() - s << endl;
#endif


    FILE *fp;
    fp = fopen("MandelbrotSet.cpp", "wb");
	fprintf(fp, "P6\n %d %d\n %d\n", Image_Width, Image_Height, MAX_PIXEL_VAL);
	fwrite(pixels, sizeof(Pixel), Image_Width * Image_Width, fp);
	fclose(fp);

	free(pixels);
	free(colors);

	return 0;
}