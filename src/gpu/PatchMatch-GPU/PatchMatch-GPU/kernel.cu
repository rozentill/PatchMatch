#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \ printf("Error at %s:%d\n",__FILE__,__LINE__); \ return EXIT_FAILURE;}} while(0)

using namespace cv;
using namespace std;


__host__ __device__ unsigned int XY_TO_INT(int x, int y) {
	return ((y) << 12) | (x);
}
__host__ __device__ int INT_TO_X(unsigned int v) {
	return (v)&((1 << 12) - 1);
}
__host__ __device__ int INT_TO_Y(unsigned int v) {
	return (v >> 12)&((1 << 12) - 1);
}

__host__ __device__ int Max(int a, int b){
	if (a>b){
		return a;
	}
	else{
		return b;
	}
}

__host__ __device__ int Min(int a, int b){
	if (a<b){
		return a;
	}
	else{
		return b;
	}
}


__host__ int dist_p(Mat a, Mat b, int ax, int ay, int bx, int by){
	Vec3b ai = a.at<Vec3b>(ay, ax);
	Vec3b bi = b.at<Vec3b>(by, bx);
	int dr = abs(ai.val[2] - bi.val[2]);
	int dg = abs(ai.val[1] - bi.val[1]);
	int db = abs(ai.val[0] - bi.val[0]);
	return dr*dr + dg*dg + db*db;
}

/* nearest voting */
__host__ Mat reconstruct(Mat a, Mat b, unsigned int * ann, int patch_w){
	Mat c;
	a.copyTo(c);
	int aew = a.cols - patch_w + 1, aeh = b.rows - patch_w + 1;       /* Effective width and height (possible upper left corners of patches). */
	int bew = b.cols - patch_w + 1, beh = b.rows - patch_w + 1;
	int ystart = 0, yend = aeh, ychange = 1;
	int xstart = 0, xend = aew, xchange = 1;
	unsigned int ybest = 0, xbest = 0, v = 0;
	//difference of pixel
	unsigned int ** pnnd;
	unsigned int ** pnn;
	pnn = new unsigned int *[a.rows];
	pnnd = new unsigned int *[a.rows];
	for (int i = 0; i < a.rows; i++)
	{
		pnn[i] = new unsigned int[a.cols];
		pnnd[i] = new unsigned int[a.cols];
		memset(pnn[i], 0, a.cols);
	}

	
	for (int ay = 0; ay < aeh; ay++) {
		for (int ax = 0; ax < aew; ax++) {
			
			if (ay < aeh&&ax < aew)
			{
				pnn[ay][ax] = ann[ay*aew+ax];
				v = ann[ay*aew + ax];
				xbest = INT_TO_X(v);
				ybest = INT_TO_Y(v);
				pnnd[ay][ax] = dist_p(a, b, ax, ay, xbest, ybest);
			}
			else if (ay >= aeh&&ax < aew){
				v = ann[(aeh-1)*aew+ax];
				xbest = INT_TO_X(v);
				ybest = INT_TO_Y(v);
				ybest += (ay - aeh+1);
				pnn[ay][ax] = XY_TO_INT(xbest, ybest);
				pnnd[ay][ax] = dist_p(a, b, ax, ay, xbest, ybest);
			}
			else if (ay < aeh&&ax >= aew){
				v = ann[ay*aew+aew-1];
				xbest = INT_TO_X(v);
				ybest = INT_TO_Y(v);
				xbest += (ax - aew+1);
				pnn[ay][ax] = XY_TO_INT(xbest, ybest);
				pnnd[ay][ax] = dist_p(a, b, ax, ay, xbest, ybest);
		
			}
			else{
				v = ann[(aeh-1)*aew+aew-1];
				xbest = INT_TO_X(v);
				ybest = INT_TO_Y(v);
				xbest += (ax - aew+1);
				ybest += (ay - aeh+1);
				pnn[ay][ax] = XY_TO_INT(xbest, ybest);
				pnnd[ay][ax] = dist_p(a, b, ax, ay, xbest, ybest);
			}

		}
	}


	for (int ay = ystart; ay != yend; ay += ychange) {
		for (int ax = xstart; ax != xend; ax += xchange) {
			v = ann[ay*aew+ax];
			xbest = INT_TO_X(v);
			ybest = INT_TO_Y(v);

			for (int dy = 0; dy < patch_w; dy++) {
				for (int dx = 0; dx < patch_w; dx++) {
					if (pnnd[ay + dy][ax + dx]>dist_p(a, b, ax + dx, ay + dy, xbest + dx ,ybest + dy)){
						pnn[ay + dy][ax + dx] = XY_TO_INT(xbest + dx, ybest + dy);
						pnnd[ay + dy][ax + dx] = dist_p(a, b, ax + dx, ay + dy, xbest + dx, ybest + dy);
					}

				}

			}
		}
	}

	for (int ay = ystart; ay < a.rows;ay++){
		for (int ax = xstart; ax < a.cols; ax++)
		{
			v = pnn[ay][ax];
			xbest = INT_TO_X(v);
			ybest = INT_TO_Y(v);

			Vec3b bi = b.at<Vec3b>(ybest, xbest);
			c.at<Vec3b>(ay, ax).val[2]  = bi.val[2];
			c.at<Vec3b>(ay, ax).val[1] = bi.val[1];
			c.at<Vec3b>(ay, ax).val[0] = bi.val[0];
		}
	}
	return c;
}


__host__ __device__ int dist(int * a, int * b, int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int bx, int by, int patch_w, int cutoff = INT_MAX){
	
	int ans = 0;
	for (int dy = 0; dy < patch_w; dy++) {
		for (int dx = 0; dx < patch_w; dx++) {
			int dr = a[(ay + dy) * 3 * a_cols + (ax + dx) * 3 + 2] - b[(by + dy) * 3 * b_cols + (bx + dx) * 3 + 2];
			int dg = a[(ay + dy) * 3 * a_cols + (ax + dx) * 3 + 1] - b[(by + dy) * 3 * b_cols + (bx + dx) * 3 + 1];
			int db = a[(ay + dy) * 3 * a_cols + (ax + dx) * 3 + 0] - b[(by + dy) * 3 * b_cols + (bx + dx) * 3 + 0];
			ans += dr*dr + dg*dg + db*db;
		}
		if (ans >= cutoff) { return cutoff; }
	}
	return ans;
}

__host__ void convertMatToArray(Mat mat, int *& arr){
	arr = new int[mat.rows*mat.cols*3];
	for (int i = 0; i < mat.rows; i++)
	{
		for (int j = 0; j <mat.cols; j++)
		{
			Vec3b rgb = mat.at<Vec3b>(i, j);
			arr[i*mat.cols*3 + j * 3 + 0] = rgb.val[0];
			arr[i*mat.cols*3 + j * 3 + 1] = rgb.val[1];
			arr[i*mat.cols*3 + j * 3 + 2] = rgb.val[2];
		}
	}
}

__host__ void convertArrayToMat(Mat & mat, int *arr){
	for (int i = 0; i < mat.rows; i++)
	{
		for (int j = 0; j <mat.cols; j++)
		{
			mat.at<Vec3b>(i, j).val[0] = arr[i*mat.cols*3 + j * 3 + 0];
			mat.at<Vec3b>(i, j).val[1] = arr[i*mat.cols*3 + j * 3 + 1];
			mat.at<Vec3b>(i, j).val[2] = arr[i*mat.cols*3 + j * 3 + 2];
		}
	}
}

__host__ void initialAnn(unsigned int *& ann, int *& annd, int aw, int ah, int bw, int bh, int aew, int aeh, int bew, int beh, int * a, int * b, int patch_w){
	for (int ay = 0; ay < aeh; ay++) {
		for (int ax = 0; ax < aew; ax++) {
			int bx = rand() % bew;
			int by = rand() % beh;

			ann[ay*aew+ax] = XY_TO_INT(bx, by);
			annd[ay*aew+ax] = dist(a, b, ah, aw, bh, bw, ax, ay, bx, by, patch_w);

		}
	}
}

__host__ void print(String string){
	cout << string;
}

__device__ void improve_guess(int * a, int * b,int a_rows, int a_cols,int b_rows, int b_cols, int ax, int ay, int &xbest, int &ybest, int dbest, int xp, int yp,int patch_w){
	int d = 0;
	d = dist(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xp, yp, patch_w, dbest);

	if (d < dbest){
		xbest = xp;
		ybest = yp;
	}
}

__device__ float cuRand(unsigned int * seed){//random number in cuda
	unsigned long a = 16807;
	unsigned long m = 2147483647;
	unsigned long x = (unsigned long)* seed;
	x = (a*x) % m;
	*seed = (unsigned int)x;
	return ((float)x / m);
}

/*get the approximate nearest neighbor and set it into ann
********************************************************
                params: 7
-----------------------------------------------
				0 - a_rows
				1 - a_cols
				2 - b_rows
				3 - b_cols
				4 - patch_w
				5 - pm_iter
				6 - rs_max
*********************************************************/
__global__ void patchmatch(int * a, int * b, unsigned int *&ann, int *&annd, int * params) {

	int ax = threadIdx.x;
	int ay = blockIdx.x;
	int aew = blockDim.x;//threads per block
	int aeh = gridDim.x;//blocks per dim

	//assign params
	int a_rows = params[0];
	int a_cols = params[1];
	int b_rows = params[2];
	int b_cols = params[3];
	int patch_w = params[4];
	int pm_iters = params[5];
	int rs_max = params[6];

	int bew = b_cols - patch_w + 1;
	int beh = b_rows - patch_w + 1;

	// for random number
	unsigned int seed = blockIdx.x*blockDim.x + threadIdx.x;

	for (int iter = 0; iter < pm_iters; iter++) {
		/* In each iteration, improve the NNF, by jumping flooding. */
		for (int jump = 8; jump > 0; jump /= 2){

			/* Current (best) guess. */
			unsigned int v = ann[ay*aew + ax];
			int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
			int dbest = annd[ay*aew + ax];

			/* Propagation: Improve current guess by trying instead correspondences from left, right, up and downs. */
			if ((ax - jump) >= 0)//left
			{
				int vp = ann[ay*aew + ax - jump];//the pixel coordinates in image b
				int xp = INT_TO_X(vp) + jump, yp = INT_TO_Y(vp);//the propagated match from vp
				if ((unsigned)xp < (unsigned)bew)
				{
					//improve guress
					improve_guess(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w);
				}
			}

			if ((ax + jump) < aew)//right
			{
				int vp = ann[ay*aew + ax + jump];//the pixel coordinates in image b
				int xp = INT_TO_X(vp) - jump, yp = INT_TO_Y(vp);
				if ((unsigned)xp >= 0)
				{
					//improve guress
					improve_guess(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w);
				}
			}

			if ((ay + jump) < aeh)//up
			{
				int vp = ann[(ay + jump)*aew + ax];//the pixel coordinates in image b
				int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) - jump;
				if ((unsigned)yp >= 0)
				{
					//improve guress
					improve_guess(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w);
				}
			}

			if ((ay - jump) >= 0)//down
			{
				int vp = ann[(ay - jump)*aew + ax];//the pixel coordinates in image b
				int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + jump;
				if ((unsigned)yp < (unsigned)beh)
				{
					//improve guress
					improve_guess(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w);
				}
			}
			__syncthreads();
			/* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
			int rs_start = rs_max;
			if (rs_start > Max(b_cols, b_rows)) { rs_start = Max(b_cols, b_rows); }
			for (int mag = rs_start; mag >= 1; mag /= 2) {
				/* Sampling window */
				int xmin = Max(xbest - mag, 0), xmax = Min(xbest + mag + 1, bew);
				int ymin = Max(ybest - mag, 0), ymax = Min(ybest + mag + 1, beh);
				int xp = xmin + (int)(cuRand(&seed)*(xmax - xmin)) % (xmax - xmin);
				int yp = ymin + (int)(cuRand(&seed)*(ymax - ymin)) % (ymax - ymin);

				improve_guess(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w);

			}
			__syncthreads();
			ann[ay*aew + ax] = XY_TO_INT(xbest, ybest);
			annd[ay*aew + ax] = dbest;
		}
	}
	
}

int main()
{
	
	/*** Definition and initialization ***/
	string window_name = "reconstructed";
	string file_a, file_b;

	Mat a, b, c;
	int * a_host, *b_host, *a_device, *b_device, *annd_host, *annd_device, *params_host, *params_device;
	int a_size, b_size, aeh, aew, beh, bew;
	unsigned int * ann_host, * ann_device;

	
	const int patch_w = 7;
	int pm_iters = 5;
	int sizeOfParams = 7;
	int rs_max = INT_MAX;
	int threadsPerBlock, blocksPerGrid;
	/*** load files ***/

	cout << "Please enter the filename of image A in the 'Image' folder :(E.g. a1.png) :";
	cin >> file_a;
	cout << "Please enter the filename of image B in the 'Image' folder :(E.g. b1.png) :";
	cin >> file_b;

	a = imread("Image/"+file_a);
	b = imread("Image/"+file_b);

	a_size = a.rows*a.cols * 3;
	b_size = b.rows*b.cols * 3;

	a.copyTo(c);//initialize c
	
	aew = a.cols - patch_w + 1; 
	aeh = a.rows - patch_w + 1;
	bew = b.cols - patch_w + 1;
	beh = b.rows - patch_w + 1;

	blocksPerGrid = aeh;
	threadsPerBlock = aew;

	//judge whether it is empty
	if (a.empty()||b.empty())
	{	
		cout << "image cannot read!" << endl;
		waitKey();
		exit;
	}

	cout << "Images loaded." << endl;


	/* initialization */
	ann_host = (unsigned int *)malloc(threadsPerBlock*blocksPerGrid*sizeof(unsigned int));
	annd_host = (int *)malloc(threadsPerBlock*blocksPerGrid*sizeof(int));
	params_host = (int *)malloc(sizeOfParams * sizeof(int));
	params_host[0] = a.rows;
	params_host[1] = a.cols;
	params_host[2] = b.rows;
	params_host[3] = b.cols;
	params_host[4] = patch_w;
	params_host[5] = pm_iters;
	params_host[6] = rs_max;

	convertMatToArray(a, a_host);
	convertMatToArray(b, b_host);
	initialAnn(ann_host, annd_host, a.cols, a.rows, b.cols, b.rows, aew, aeh, bew, beh, a_host, b_host, patch_w);

	cudaMalloc(&a_device, a_size*sizeof(int));
	cudaMalloc(&b_device, b_size*sizeof(int));
	cudaMalloc(&annd_device, threadsPerBlock*blocksPerGrid*sizeof(int));
	cudaMalloc(&ann_device, threadsPerBlock*blocksPerGrid*sizeof(unsigned int));
	cudaMalloc(&params_device, sizeOfParams*sizeof(int));

	cudaMemcpy(a_device, a_host, a_size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b_device, b_host, b_size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(ann_device, ann_host, threadsPerBlock*blocksPerGrid*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(annd_device, annd_host, threadsPerBlock*blocksPerGrid*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(params_device, params_host, sizeOfParams*sizeof(int), cudaMemcpyHostToDevice);

	print("Now begin to find ann.\n");

	patchmatch<<<blocksPerGrid, threadsPerBlock>>>(a_device, b_device, ann_device, annd_device, params_device);
	
	print((string)"Finished finding ann.\n" + (string)"Now begin to reconstruct new image.");
	 
	cudaMemcpy(a_host, a_device, a_size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b_host, b_device, b_size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(ann_host, ann_device, threadsPerBlock*blocksPerGrid*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(annd_host, annd_device, threadsPerBlock*blocksPerGrid*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(params_host, params_device, sizeOfParams*sizeof(int), cudaMemcpyHostToDevice);

	cudaFree(a_device);
	cudaFree(b_device);
	cudaFree(ann_device);
	cudaFree(annd_device);
	cudaFree(params_device);

	c = reconstruct(a, b, ann_host,patch_w);
	cout << "Finish reconstruction of the new image." << endl;

	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	imshow(window_name, c);

	namedWindow("a", CV_WINDOW_AUTOSIZE);
	imshow("a", a);

	namedWindow("b", CV_WINDOW_AUTOSIZE);
	imshow("b", b);

	//imwrite("Image/result.png", c);

	waitKey();

	destroyAllWindows();

    return 0;
}