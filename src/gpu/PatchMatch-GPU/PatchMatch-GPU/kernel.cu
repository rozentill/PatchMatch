#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

const int patch_w = 5;
int	pm_iters = 5;
int rs_max = INT_MAX;

dim3 threadsPerBlock(patch_w,patch_w);

#define XY_TO_INT(x, y) (((y)<<12)|(x))
#define INT_TO_X(v) ((v)&((1<<12)-1))
#define INT_TO_Y(v) ((v>>12)&((1<<12)-1))

//l2 distance between two patches
__global__ void dist_gpu(cudaPitchedPtr a, cudaPitchedPtr b, int * params){//params : 0 - ax, 1 - ay, 2 - bx, 3 - by, 4 - res,
	//__shared__ int res[patch_w*patch_w];
	int dr = a.ptr[params[1] + threadIdx.y][params[0] + threadIdx.x][2] - b[by + threadIdx.y][bx + threadIdx.x][2];
	int dg = a.ptr[params[1] + threadIdx.y][params[0] + threadIdx.x][1] - b[by + threadIdx.y][bx + threadIdx.x][1];
	int db = a.ptr[params[1] + threadIdx.y][params[0] + threadIdx.x][0] - b[by + threadIdx.y][bx + threadIdx.x][0];
	/*patchd[threadIdx.y][threadIdx.x] = dr*dr + dg*dg + db*db;*/
	res[threadIdx.y*patch_w+threadIdx.x] = dr*dr + dg*dg + db*db;
	//__syncthreads();
	//int i = patch_w*patch_w / 2;
	//int j = patch_w*patch_w % 2;
	//while (i != 0)
	//{
	//	if (threadIdx.y*patch_w + threadIdx.x<i){
	//		res[threadIdx.y*patch_w + threadIdx.x] += res[threadIdx.y*patch_w + threadIdx.x + i];
	//	}
	//	if (j == 1 && threadIdx.y*patch_w + threadIdx.x == i - 1){
	//		res[threadIdx.y*patch_w + threadIdx.x] += res[threadIdx.y*patch_w + threadIdx.x + i + 1];
	//	}
	//	__syncthreads();
	//	j = i % 2;
	//	i = i / 2;
	//}
	//if (threadIdx.x == 0 && threadIdx.y == 0){
	//	//cout << "Total result is " << res[threadIdx.y*patch_w + threadIdx.x] << endl;
	//	if (res[threadIdx.y*patch_w + threadIdx.x] >= cutoff){
	//		ans = cutoff;
	//	}
	//	else{
	//		ans = res[threadIdx.y*patch_w + threadIdx.x];
	//	}
	//}
	/*ans = 0;
	for (int dy = 0; dy < patch_w; dy++) {
		for (int dx = 0; dx < patch_w; dx++) {
			int dr = a[ay + dy][ax + dx][2] - b[by + dy][bx + dx][2];
			int dg = a[ay + dy][ax + dx][1] - b[by + dy][bx + dx][1];
			int db = a[ay + dy][ax + dx][0] - b[by + dy][bx + dx][0];
			ans += dr*dr + dg*dg + db*db;
		}
		if (ans >= cutoff) { return cutoff; }
	}
	return ans;*/
	
}

int dist_test(int ***a, int ***b, int ax, int ay, int bx, int by, int cutoff = INT_MAX){
	int ans = 0;
	for (int dy = 0; dy < patch_w; dy++) {
		for (int dx = 0; dx < patch_w; dx++) {
			int dr = a[ay + dy][ax + dx][2] - b[by + dy][bx + dx][2];
			int dg = a[ay + dy][ax + dx][1] - b[by + dy][bx + dx][1];
			int db = a[ay + dy][ax + dx][0] - b[by + dy][bx + dx][0];
			ans += dr*dr + dg*dg + db*db;
		}
		if (ans >= cutoff) { return cutoff; }
	}
	return ans;
}

//int dist(Mat a, Mat b, int ax, int ay, int bx, int by, int cutoff = INT_MAX){
//	int ans = 0;
//	for (int dy = 0; dy < patch_w; dy++) {
//		for (int dx = 0; dx < patch_w; dx++) {
//			Vec3b ai = a.at<Vec3b>(ay + dy, ax + dx);
//			Vec3b bi = b.at<Vec3b>(by + dy, bx + dx);
//			int dr = abs(ai.val[2] - bi.val[2]);
//			int dg = abs(ai.val[1] - bi.val[1]);
//			int db = abs(ai.val[0] - bi.val[0]);
//			ans += dr*dr + dg*dg + db*db;
//		}
//		if (ans >= cutoff) { return cutoff; }
//	}
//	return ans;
//}

void improve_guess(int *** a, int *** b, int ax, int ay, int &xbest, int &ybest, int &dbest, int bx, int by) {
	int d_gpu[patch_w*patch_w] = { 0 }, d_cpu = 0;
	dist_gpu<<<1, threadsPerBlock >>>(a, b, ax, ay, bx, by, d_gpu, dbest);
	d_cpu = dist_test(a, b, ax, ay, bx, by, dbest);
	int resgpu = 0;
	for (int i = 0; i < patch_w; i++)
	{
		resgpu += d_gpu[i];
	}
	cout << "d-gpu is :" << resgpu << ", d-cpu:" << d_cpu << endl;
	//int d = dist_test(a, b, ax, ay, bx, by, dbest);
	if (d_cpu < dbest) {
		dbest = d_cpu;
		xbest = bx;
		ybest = by;
	}
}

//get the approximate nearest neighbor and set it into ann
void patchmatch(Mat a, Mat b, unsigned int **&ann, int **&annd) {
	
	/* Initialize with random nearest neighbor field (NNF). */
	int ans = 0;
	int aew = a.cols - patch_w + 1, aeh = b.rows - patch_w + 1;       /* Effective width and height (possible upper left corners of patches). */
	int bew = b.cols - patch_w + 1, beh = b.rows - patch_w + 1;

	int *** a_pixel = new int **[a.rows];//set the rgb value from matrix a in a_pixel
	int *** b_pixel = new int **[b.rows];
	int *** dev_a;// device variable of a_pixel
	int *** dev_b;

	int params[5] = { 0 }; // 0 - ax, 1 - ay, 2 - bx, 3 - by, 4 - res
	int * dev_params;

	ann = new unsigned int *[a.rows];
	annd = new int *[a.rows];

	// initialize ann, annd ,a_pixel, b_pixel
	for (int i = 0; i < a.rows; i++)
	{
		ann[i] = new unsigned int[a.cols];
		annd[i] = new int[a.cols];
		a_pixel[i] = new int*[a.cols];
		memset(ann[i], 0, a.cols);
		memset(annd[i], 0, a.cols);
		for (int j = 0; j < a.cols; j++)
		{
			Vec3b ai = a.at<Vec3b>(i, j);
			a_pixel[i][j] = new int[3];
			a_pixel[i][j][0] = (int)ai.val[0];
			a_pixel[i][j][1] = (int)ai.val[1];
			a_pixel[i][j][2] = (int)ai.val[2];
		}
	}
	for (int i = 0; i < b.rows; i++)
	{
		b_pixel[i] = new int*[b.cols];
		for (int j = 0; j < b.cols; j++)
		{
			Vec3b bi = b.at<Vec3b>(i, j);
			b_pixel[i][j] = new int[3];
			b_pixel[i][j][0] = (int)bi.val[0];
			b_pixel[i][j][1] = (int)bi.val[1];
			b_pixel[i][j][2] = (int)bi.val[2];
		}
	}

	// cuda malloc
	cudaError cudaStatus;
	//3 dims
	cudaExtent a_extent = make_cudaExtent(a.cols, a.rows, 3);
	cudaExtent b_extent = make_cudaExtent(b.cols, b.rows, 3);
	cudaPitchedPtr a_devPitchedPtr;
	cudaPitchedPtr b_devPitchedPtr;

	cudaMalloc3D(&a_devPitchedPtr, a_extent);
	cudaMemcpy3DParms a_HostToDev = { 0 };
	a_HostToDev.srcPtr = make_cudaPitchedPtr((void*)a_pixel, a.cols * sizeof(int), a.cols, a.rows);
	a_HostToDev.dstPtr = a_devPitchedPtr;
	a_HostToDev.extent = a_extent;
	a_HostToDev.kind = cudaMemcpyHostToDevice;
	cudaStatus = cudaMemcpy3D(&a_HostToDev);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "MemcpyHtD: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaMalloc3D(&b_devPitchedPtr, b_extent);
	cudaMemcpy3DParms b_HostToDev = { 0 };
	b_HostToDev.srcPtr = make_cudaPitchedPtr((void*)b_pixel, b.cols * sizeof(int), b.cols, b.rows);
	b_HostToDev.dstPtr = b_devPitchedPtr;
	b_HostToDev.extent = b_extent;
	b_HostToDev.kind = cudaMemcpyHostToDevice;
	cudaStatus = cudaMemcpy3D(&b_HostToDev);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "MemcpyHtD: %s\n", cudaGetErrorString(cudaStatus));
	}

	

	for (int ay = 0; ay < aeh; ay++) {
		for (int ax = 0; ax < aew; ax++) {
			int bx = rand() % bew;
			int by = rand() % beh;
			
			params[0] = ax;
			params[1] = ay;
			params[2] = bx;
			params[3] = by;

			cudaMalloc((void**)&dev_params, 5 * sizeof(int));
			cudaMemcpy(dev_params, params, 5 * sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess){
				fprintf(stderr, "MemcpyHtD: %s\n", cudaGetErrorString(cudaStatus));
			}
			dist_gpu<<<1, threadsPerBlock>>>(dev_a, dev_b, dev_params);

			cudaStatus = cudaMemcpy(params, dev_params, 5 * sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess){
				fprintf(stderr, "MemcpyDtH: %s\n", cudaGetErrorString(cudaStatus));
			}
			ann[ay][ax] = XY_TO_INT(bx, by);
			annd[ay][ax] = dev_params[4];

			cudaFree(dev_params);

			//annd[ay][ax] = dist_test(a_pixel, b_pixel , ax, ay, bx, by);
			//cout << "ann :" << ann[ay][ax] << ", annd :" << annd[ay][ax]<<endl;
			ans = 0;
			
			
		}
	}
	for (int iter = 0; iter < pm_iters; iter++) {
		/* In each iteration, improve the NNF, by looping in scanline or reverse-scanline order. */
		int ystart = 0, yend = aeh, ychange = 1;
		int xstart = 0, xend = aew, xchange = 1;
		if (iter % 2 == 1) {
			xstart = xend - 1; xend = -1; xchange = -1;
			ystart = yend - 1; yend = -1; ychange = -1;
		}
		for (int ay = ystart; ay != yend; ay += ychange) {
			for (int ax = xstart; ax != xend; ax += xchange) {
				/* Current (best) guess. */
				unsigned int v = ann[ay][ax];
				int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
				int dbest = annd[ay][ax];

				/* Propagation: Improve current guess by trying instead correspondences from left and above (below and right on odd iterations). */
				if ((unsigned)(ax - xchange) < (unsigned)aew && (ax - xchange) >= 0) {
					int vp = ann[ay][ax - xchange];
					int xp = INT_TO_X(vp) + xchange, yp = INT_TO_Y(vp);
					if ((unsigned)xp < (unsigned)bew) {
						//improve guress
						params[0] = ax;
						params[1] = ay;
						params[2] = xp;
						params[3] = yp;

						cudaMalloc((void**)&dev_params, 5 * sizeof(int));
						cudaMemcpy(dev_params, params, 5 * sizeof(int), cudaMemcpyHostToDevice);
						if (cudaStatus != cudaSuccess){
							fprintf(stderr, "MemcpyHtD: %s\n", cudaGetErrorString(cudaStatus));
						}
						dist_gpu<<<1, threadsPerBlock>>>(dev_a, dev_b, dev_params);

						cudaStatus = cudaMemcpy(params, dev_params, 5 * sizeof(int), cudaMemcpyDeviceToHost);
						if (cudaStatus != cudaSuccess){
							fprintf(stderr, "MemcpyDtH: %s\n", cudaGetErrorString(cudaStatus));
						}
						if (dev_params[4] < dbest){
							xbest = xp;
							ybest = yp;
							dbest = dev_params[4];
						}

						cudaFree(dev_params);

						//improve_guess(a_pixel, b_pixel, ax, ay, xbest, ybest, dbest, xp, yp);
					}
				}

				if ((unsigned)(ay - ychange) < (unsigned)aeh && (ay - ychange) >= 0) {
					int vp = ann[ay - ychange][ax];
					int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + ychange;
					if ((unsigned)yp < (unsigned)beh) {
						//improve guress
						params[0] = ax;
						params[1] = ay;
						params[2] = xp;
						params[3] = yp;

						cudaMalloc((void**)&dev_params, 5 * sizeof(int));
						cudaMemcpy(dev_params, params, 5 * sizeof(int), cudaMemcpyHostToDevice);
						if (cudaStatus != cudaSuccess){
							fprintf(stderr, "MemcpyHtD: %s\n", cudaGetErrorString(cudaStatus));
						}
						dist_gpu<<<1, threadsPerBlock>>>(dev_a, dev_b, dev_params);

						cudaStatus = cudaMemcpy(params, dev_params, 5 * sizeof(int), cudaMemcpyDeviceToHost);
						if (cudaStatus != cudaSuccess){
							fprintf(stderr, "MemcpyDtH: %s\n", cudaGetErrorString(cudaStatus));
						}
						if (dev_params[4] < dbest){
							xbest = xp;
							ybest = yp;
							dbest = dev_params[4];
						}

						cudaFree(dev_params);
						//improve_guess(a_pixel, b_pixel, ax, ay, xbest, ybest, dbest, xp, yp);
					}
				}

				/* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
				int rs_start = rs_max;
				if (rs_start > MAX(b.cols, b.rows)) { rs_start = MAX(b.cols, b.rows); }
				for (int mag = rs_start; mag >= 1; mag /= 2) {
					/* Sampling window */
					int xmin = MAX(xbest - mag, 0), xmax = MIN(xbest + mag + 1, bew);
					int ymin = MAX(ybest - mag, 0), ymax = MIN(ybest + mag + 1, beh);
					int xp = xmin + rand() % (xmax - xmin);
					int yp = ymin + rand() % (ymax - ymin);
					//improve_guess(a_pixel, b_pixel, ax, ay, xbest, ybest, dbest, xp, yp);
					//improve guress
					params[0] = ax;
					params[1] = ay;
					params[2] = xp;
					params[3] = yp;

					cudaMalloc((void**)&dev_params, 5 * sizeof(int));
					cudaMemcpy(dev_params, params, 5 * sizeof(int), cudaMemcpyHostToDevice);
					if (cudaStatus != cudaSuccess){
						fprintf(stderr, "MemcpyHtD: %s\n", cudaGetErrorString(cudaStatus));
					}
					dist_gpu<<<1, threadsPerBlock>>>(dev_a, dev_b, dev_params);

					cudaStatus = cudaMemcpy(params, dev_params, 5 * sizeof(int), cudaMemcpyDeviceToHost);
					if (cudaStatus != cudaSuccess){
						fprintf(stderr, "MemcpyDtH: %s\n", cudaGetErrorString(cudaStatus));
					}
					if (dev_params[4] < dbest){
						xbest = xp;
						ybest = yp;
						dbest = dev_params[4];
					}

					cudaFree(dev_params);
				}

				ann[ay][ax] = XY_TO_INT(xbest, ybest);
				annd[ay][ax] = dbest;
			}
		}
	}
}

int dist_p(Mat a, Mat b, int ax, int ay, int bx, int by){
	Vec3b ai = a.at<Vec3b>(ay, ax);
	Vec3b bi = b.at<Vec3b>(by, bx);
	int dr = abs(ai.val[2] - bi.val[2]);
	int dg = abs(ai.val[1] - bi.val[1]);
	int db = abs(ai.val[0] - bi.val[0]);
	return dr*dr + dg*dg + db*db;
}

/* nearest voting */
Mat reconstruct(Mat a, Mat b, unsigned int ** ann){
	Mat a_recon;
	a.copyTo(a_recon);
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
				pnn[ay][ax] = ann[ay][ax];
				v = ann[ay][ax];
				xbest = INT_TO_X(v);
				ybest = INT_TO_Y(v);
				pnnd[ay][ax] = dist_p(a, b, ax, ay, xbest, ybest);
			}
			else if (ay >= aeh&&ax < aew){
				v = ann[aeh-1][ax];
				xbest = INT_TO_X(v);
				ybest = INT_TO_Y(v);
				ybest += (ay - aeh+1);
				pnn[ay][ax] = XY_TO_INT(xbest, ybest);
				pnnd[ay][ax] = dist_p(a, b, ax, ay, xbest, ybest);
			}
			else if (ay < aeh&&ax >= aew){
				v = ann[ay][aew-1];
				xbest = INT_TO_X(v);
				ybest = INT_TO_Y(v);
				xbest += (ax - aew+1);
				pnn[ay][ax] = XY_TO_INT(xbest, ybest);
				pnnd[ay][ax] = dist_p(a, b, ax, ay, xbest, ybest);
		
			}
			else{
				v = ann[aeh-1][aew-1];
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
			v = ann[ay][ax];
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
			a_recon.at<Vec3b>(ay, ax).val[2] = bi.val[2];
			a_recon.at<Vec3b>(ay, ax).val[1] = bi.val[1];
			a_recon.at<Vec3b>(ay, ax).val[0] = bi.val[0];
		}
	}
	return a_recon;
}

//
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}

int main()
{

	String window_name = "reconstructed";

	// define img matrix
	Mat a = imread("Image/disp1.png");
	Mat b = imread("Image/view1.png");
	Mat a_recon;
	if (a.empty()||b.empty())
	{	
		cout << "image cannot read!" << endl;
		waitKey();
		exit;
	}
	// define and initialize ann and annd array
	int **annd;
	unsigned int**ann;
	patchmatch(a, b, ann, annd);

	a_recon = reconstruct(a, b, ann);

	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	imshow(window_name, a_recon);

	namedWindow("a", CV_WINDOW_AUTOSIZE);
	imshow("a", a);

	namedWindow("b", CV_WINDOW_AUTOSIZE);
	imshow("b", b);

	//imwrite("Image/result.png", a_recon);

	waitKey();

	destroyAllWindows();

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
