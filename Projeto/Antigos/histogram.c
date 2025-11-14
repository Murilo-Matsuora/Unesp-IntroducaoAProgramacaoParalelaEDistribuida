#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>
#include <omp.h>

#define RGB_COMPONENT_COLOR 255
#define DIVS 8

typedef struct
{
	unsigned char red, green, blue;
} PPMPixel;

typedef struct
{
	int x, y;
	PPMPixel *data;
} PPMImage;

static PPMImage *readPPM(const char *filename)
{
	char buff[16];
	PPMImage *img;
	FILE *fp;
	int c, rgb_comp_color;

	// Open file in binary mode
	fp = fopen(filename, "rb");
	if (!fp)
	{
		perror("Cannot open file");
		exit(1);
	}

	if (!fgets(buff, sizeof(buff), fp))
	{
		perror("File read error");
		exit(1);
	}

	if (buff[0] != 'P' || buff[1] != '6')
	{
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	img = (PPMImage *)malloc(sizeof(PPMImage));
	if (!img)
	{
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	c = getc(fp);
	while (c == '#')
	{ // Skip comments
		while (getc(fp) != '\n')
			;
		c = getc(fp);
	}

	ungetc(c, fp);
	if (fscanf(fp, "%d %d", &img->x, &img->y) != 2)
	{
		fprintf(stderr, "Invalid image size (error loading)\n");
		exit(1);
	}

	if (fscanf(fp, "%d", &rgb_comp_color) != 1)
	{
		fprintf(stderr, "Invalid rgb component (error loading)\n");
		exit(1);
	}

	if (rgb_comp_color != RGB_COMPONENT_COLOR)
	{
		fprintf(stderr, "Image does not have 8-bits components\n");
		exit(1);
	}

	while (fgetc(fp) != '\n')
		;
	img->data = (PPMPixel *)malloc(img->x * img->y * sizeof(PPMPixel));
	if (!img->data)
	{
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	if (fread(img->data, 3 * img->x, img->y, fp) != img->y)
	{
		fprintf(stderr, "Error loading image.\n");
		exit(1);
	}

	fclose(fp);
	return img;
}

void Histogram(PPMImage *image, float *h) {
    int i, j, k, l;
    long n = image->x * image->y;

    
    

    // Step 2: Initialize the histogram
    #pragma omp parallel for
    for (i = 0; i < DIVS * DIVS * DIVS; i++) {
        h[i] = 0.0;
    }

    #pragma omp parallel
    {
        float local_h[DIVS * DIVS * DIVS] = {0}; // Thread-local histogram

        // Each thread processes bins independently
        #pragma omp for collapse(3)
        for (j = 0; j < DIVS; j++) {
            for (k = 0; k < DIVS; k++) {
                for (l = 0; l < DIVS; l++) {
                    int count = 0;

                    // Compare all pixels against the current bin
                    for (i = 0; i < n; i++) {
                        if (image->data[i].red == j && 
                            image->data[i].green == k && 
                            image->data[i].blue == l) {
                            count++;
                        }
                    }

                    local_h[j * DIVS * DIVS + k * DIVS + l] = count;
                }
            }
        }

        // Merge thread-local histograms into the global histogram
        #pragma omp critical
        {
            for (int idx = 0; idx < DIVS * DIVS * DIVS; idx++) {
                h[idx] += local_h[idx];
            }
        }
    }

    // Step 4: Normalize the histogram
    for (i = 0; i < DIVS * DIVS * DIVS; i++) {
        h[i] = h[i] / n;
    }
}

void measure_execution_time(PPMImage *image, float *h, int num_iterations) {
    // LARGE_INTEGER frequency; // Frequency of the high-resolution counter
    // LARGE_INTEGER start, end; // Start and end timestamps
    double total_time = 0.0;

    // Get the frequency of the high-resolution counter
    // QueryPerformanceFrequency(&frequency);

    for (int iter = 0; iter < num_iterations; iter++) {
        printf("Starting iteration %d...\n", iter + 1);
 
		// QueryPerformanceCounter(&start);

        // // Call the function you want to measure (e.g., Histogram)
        // Histogram(image, h);

        // // Record the end time
        // QueryPerformanceCounter(&end);

        // // Calculate the elapsed time in seconds
        // double elapsed_time = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;

		double start_time = omp_get_wtime();
		Histogram(image, h);
		double end_time = omp_get_wtime();

		double elapsed_time = end_time - start_time;
       
        printf("Iteration %d took %0.8f seconds.\n", iter + 1, elapsed_time);

        // Add to total time
        total_time += elapsed_time;
    }

    // Calculate and display the average time per iteration
    double average_time = total_time / num_iterations;
    printf("\nTotal time for %d iterations: %0.8f seconds.\n", num_iterations, total_time);
    printf("Average time per iteration: %0.8f seconds.\n", average_time);
}

int main(int argc, char *argv[])
{

	int i;

	printf("3\n");

	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s <image_file>\n", argv[0]);
		return 1;
	}

	PPMImage *image = readPPM(argv[1]);

	float *h = (float *)malloc(sizeof(float) * DIVS * DIVS * DIVS);

	long n = image->x * image->y;

	printf("NUMERO DE THREADS: %d\n",omp_get_max_threads());

	// Step 1: Normalize pixel values (parallelized)
	#pragma omp parallel for
    for (i = 0; i < n; i++) {
        image->data[i].red = floor((image->data[i].red * DIVS) / 256);
        image->data[i].green = floor((image->data[i].green * DIVS) / 256);
        image->data[i].blue = floor((image->data[i].blue * DIVS) / 256);
    }

	int num_iterations = 40;
    measure_execution_time(image, h, num_iterations);

	for (i = 0; i < DIVS * DIVS * DIVS; i++)
	{
		if (h[i] != 0)
			printf("(%d, %d, %d): %f\n", i / (DIVS * DIVS), (i / DIVS) % (DIVS), i % DIVS, h[i]);

	}
	printf("\n");
	free(h);
    free(image->data);
    free(image);

	return 0;
}
