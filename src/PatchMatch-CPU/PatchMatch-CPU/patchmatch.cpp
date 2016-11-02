#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include "time.h"


using namespace cv;
using namespace std;

const int patch_w = 3;
int	pm_iters = 5;
int rs_max = INT_MAX;

#define XY_TO_INT(x, y) (((y)<<12)|(x))
#define INT_TO_X(v) ((v)&((1<<12)-1))
#define INT_TO_Y(v) ((v>>12)&((1<<12)-1))

//l2 distance between two patches
float dist(int ***a, int ***b, int ax, int ay, int bx, int by,int a_rows, int a_cols, int b_rows, int b_cols, float cutoff = INT_MAX) {
	float ans = 0, num = 0;
	for (int dy = -patch_w/2; dy <= patch_w/2; dy++) {
		for (int dx = -patch_w/2; dx <= patch_w/2; dx++) {
			if (
				(ay + dy) < a_rows && (ay + dy) >= 0 && (ax + dx) < a_cols && (ax + dx) >= 0
				&&
				(by + dy) < b_rows && (by + dy) >= 0 && (bx + dx) < b_cols && (bx + dx) >= 0
				)//the pixel in a should exist and pixel in b should exist
			{
				int dr = a[ay + dy][ax + dx][2] - b[by + dy][bx + dx][2];
				int dg = a[ay + dy][ax + dx][1] - b[by + dy][bx + dx][1];
				int db = a[ay + dy][ax + dx][0] - b[by + dy][bx + dx][0];
				ans += dr*dr + dg*dg + db*db;
				num += 1;
			}
			
		}
		
	}
	ans = ans / num;
	if (ans >= cutoff) { return cutoff; }
	else {
		return ans;
	}
	
}

void improve_guess(int *** a, int *** b, int ax, int ay, int &xbest, int &ybest, float &dbest, int bx, int by , int a_rows, int a_cols, int b_rows, int b_cols) {
	float d_cpu = 0;

	d_cpu = dist(a, b, ax, ay, bx, by, a_rows, a_cols, b_rows, b_cols,dbest);
	if (d_cpu < dbest) {
		dbest = d_cpu;
		xbest = bx;
		ybest = by;
	}
}

//get the approximate nearest neighbor and set it into ann
void patchmatch(Mat a, Mat b, unsigned int **&ann, float **&annd) {

	/* Initialize with random nearest neighbor field (NNF). */
	int ans = 0;

	int *** a_pixel = new int **[a.rows];//set the rgb value from matrix a in a_pixel
	int *** b_pixel = new int **[b.rows];
	ann = new unsigned int *[a.rows];
	annd = new float *[a.rows];

	// initialize ann, annd ,a_pixel, b_pixel
	for (int i = 0; i < a.rows; i++)
	{
		ann[i] = new unsigned int[a.cols];
		annd[i] = new float[a.cols];
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

	for (int ay = 0; ay < a.rows; ay++) {
		for (int ax = 0; ax < a.cols; ax++) {
			int bx = rand() % b.cols;
			int by = rand() % b.rows;

			ann[ay][ax] = XY_TO_INT(bx, by);
			annd[ay][ax] = dist(a_pixel, b_pixel, ax, ay, bx, by,a.rows,a.cols,b.rows,b.cols);

		}
	}
	//cout << "Finished randomly choosing nearest neighbor patches." << endl;
	//cout << "Now begin iteration..." << endl;
	for (int iter = 0; iter < pm_iters; iter++) {
		//cout << "The iteration " << iter + 1 << "..." << endl;
		/* In each iteration, improve the NNF, by looping in scanline or reverse-scanline order. */
		int ystart = 0, yend = a.rows, ychange = 1;
		int xstart = 0, xend = a.cols, xchange = 1;
		if (iter % 2 == 1) {
			xstart = xend - 1; xend = -1; xchange = -1;
			ystart = yend - 1; yend = -1; ychange = -1;
		}
		for (int ay = ystart; ay != yend; ay += ychange) {
			for (int ax = xstart; ax != xend; ax += xchange) {
				/* Current (best) guess. */
				unsigned int v = ann[ay][ax];
				int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
				float dbest = annd[ay][ax];

				/* Propagation: Improve current guess by trying instead correspondences from left and above (below and right on odd iterations). */
				if ((unsigned)(ax - xchange) < (unsigned)a.cols && (ax - xchange) >= 0) {
					int vp = ann[ay][ax - xchange];
					int xp = INT_TO_X(vp) + xchange, yp = INT_TO_Y(vp);
					if ((unsigned)xp < (unsigned)b.cols) {

						improve_guess(a_pixel, b_pixel, ax, ay, xbest, ybest, dbest, xp, yp, a.rows, a.cols, b.rows, b.cols);
					}
				}

				if ((unsigned)(ay - ychange) < (unsigned)a.rows && (ay - ychange) >= 0) {
					int vp = ann[ay - ychange][ax];
					int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + ychange;
					if ((unsigned)yp < (unsigned)b.rows) {

						improve_guess(a_pixel, b_pixel, ax, ay, xbest, ybest, dbest, xp, yp, a.rows, a.cols, b.rows, b.cols);

					}
				}

				/* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
				int rs_start = rs_max;
				if (rs_start > MAX(b.cols, b.rows)) { rs_start = MAX(b.cols, b.rows); }
				for (int mag = rs_start; mag >= 1; mag /= 2) {
					/* Sampling window */
					int xmin = MAX(xbest - mag, 0), xmax = MIN(xbest + mag + 1, b.cols);
					int ymin = MAX(ybest - mag, 0), ymax = MIN(ybest + mag + 1, b.rows);
					int xp = xmin + rand() % (xmax - xmin);
					int yp = ymin + rand() % (ymax - ymin);

					improve_guess(a_pixel, b_pixel, ax, ay, xbest, ybest, dbest, xp, yp, a.rows, a.cols, b.rows, b.cols);

				}

				ann[ay][ax] = XY_TO_INT(xbest, ybest);
				annd[ay][ax] = dbest;
			}
		}
	}

}

int dist_p(Mat a, Mat b, int ax, int ay, int bx, int by) {
	Vec3b ai = a.at<Vec3b>(ay, ax);
	Vec3b bi = b.at<Vec3b>(by, bx);
	int dr = abs(ai.val[2] - bi.val[2]);
	int dg = abs(ai.val[1] - bi.val[1]);
	int db = abs(ai.val[0] - bi.val[0]);
	return dr*dr + dg*dg + db*db;
}

Mat reconstruct_flow(Mat a, Mat b, unsigned int ** ann, int patch_w) {
	Mat c;
	a.copyTo(c);
	for (int ay = 0; ay < a.rows; ay++) {
		for (int ax = 0; ax < a.cols; ax++)
		{
			unsigned int v = ann[ay][ax];
			int xbest = INT_TO_X(v);
			int ybest = INT_TO_Y(v);

			Vec3b bi = b.at<Vec3b>(ybest, xbest);
			c.at<Vec3b>(ay, ax).val[0] = (uchar)255 * ((float)xbest / b.cols);
			c.at<Vec3b>(ay, ax).val[2] = 0;
			c.at<Vec3b>(ay, ax).val[1] = (uchar)255 * ((float)ybest / b.rows);
		}
	}
	return c;
}

/* nearest voting */
Mat reconstruct_nearest(Mat a, Mat b, unsigned int ** ann) {
	Mat a_recon;
	a.copyTo(a_recon);
	
	int ystart = 0, yend = a.rows, ychange = 1;
	int xstart = 0, xend = a.cols, xchange = 1;
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


	for (int ay = 0; ay < a.rows; ay++) {
		for (int ax = 0; ax < a.cols; ax++) {

			if (ay < a.rows&&ax < a.cols)
			{
				pnn[ay][ax] = ann[ay][ax];
				v = ann[ay][ax];
				xbest = INT_TO_X(v);
				ybest = INT_TO_Y(v);
				pnnd[ay][ax] = dist_p(a, b, ax, ay, xbest, ybest);
			}
			else if (ay >= a.rows&&ax < a.cols) {
				v = ann[a.rows - 1][ax];
				xbest = INT_TO_X(v);
				ybest = INT_TO_Y(v);
				ybest += (ay - a.rows + 1);
				pnn[ay][ax] = XY_TO_INT(xbest, ybest);
				pnnd[ay][ax] = dist_p(a, b, ax, ay, xbest, ybest);
			}
			else if (ay < a.rows&&ax >= a.cols) {
				v = ann[ay][a.cols - 1];
				xbest = INT_TO_X(v);
				ybest = INT_TO_Y(v);
				xbest += (ax - a.cols + 1);
				pnn[ay][ax] = XY_TO_INT(xbest, ybest);
				pnnd[ay][ax] = dist_p(a, b, ax, ay, xbest, ybest);

			}
			else {
				v = ann[a.rows - 1][a.cols - 1];
				xbest = INT_TO_X(v);
				ybest = INT_TO_Y(v);
				xbest += (ax - a.cols + 1);
				ybest += (ay - a.rows + 1);
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
					if (pnnd[ay + dy][ax + dx]>dist_p(a, b, ax + dx, ay + dy, xbest + dx, ybest + dy)) {
						pnn[ay + dy][ax + dx] = XY_TO_INT(xbest + dx, ybest + dy);
						pnnd[ay + dy][ax + dx] = dist_p(a, b, ax + dx, ay + dy, xbest + dx, ybest + dy);
					}

				}

			}
		}
	}

	for (int ay = ystart; ay < a.rows; ay++) {
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


Mat reconstruct_center(Mat a, Mat b, unsigned int ** ann) {
	Mat a_recon;
	a.copyTo(a_recon);

	int ystart = 0, yend = a.rows, ychange = 1;
	int xstart = 0, xend = a.cols, xchange = 1;
	for (int ay = ystart; ay < a.rows; ay++) {
		for (int ax = xstart; ax < a.cols; ax++)
		{
			int v = ann[ay][ax];
			int xbest = INT_TO_X(v);
			int ybest = INT_TO_Y(v);

			Vec3b bi = b.at<Vec3b>(ybest, xbest);
			a_recon.at<Vec3b>(ay, ax).val[2] = bi.val[2];
			a_recon.at<Vec3b>(ay, ax).val[1] = bi.val[1];
			a_recon.at<Vec3b>(ay, ax).val[0] = bi.val[0];
		}
	}
	return a_recon;
}

int main()
{

	clock_t start, finish;
	double duration;

	string window_name = "reconstructed";
	//string file_a, file_b;

	//cout << "Please enter the filename of image A in the 'Image' folder :(E.g. a1.png) :";
	//cin >> file_a;
	//cout << "Please enter the filename of image B in the 'Image' folder :(E.g. b1.png) :";
	//cin >> file_b;

	string file_a = "bike_a.png";
	string file_b = "bike_b.png";
	string image_path = "D:/MSRA/Code/Project/Artistic-Filter/Artistic Filter/Artistic Filter/files/images/";

	// define img matrix
	Mat a = imread(image_path + file_a);
	Mat b = imread(image_path + file_b);
	Mat a_recon;
	if (a.empty() || b.empty())
	{
		cout << "image cannot read!" << endl;
		waitKey();
		return;
	}

	cout << "Images loaded." << endl;
	// define and initialize ann and annd array
	float **annd;
	unsigned int**ann;

	cout << "Now begin to find ann." << endl;
	start = clock();
	patchmatch(a, b, ann, annd);
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "Finished finding ann. Time : "<<duration<<" seconds." << endl;
	cout << "Now begin to reconstruct new image." << endl;

	a_recon = reconstruct_center(a, b, ann);
	
	cout << "Finish reconstruction of the new image." << endl;

	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	imshow(window_name, a_recon);

	Mat flowmap = reconstruct_flow(a, b, ann, patch_w);
	namedWindow("flow", CV_WINDOW_AUTOSIZE);
	imshow("flow", flowmap);

	namedWindow("a", CV_WINDOW_AUTOSIZE);
	imshow("a", a);

	namedWindow("b", CV_WINDOW_AUTOSIZE);
	imshow("b", b);

	imwrite(image_path+"bike_flow_cpu.png", flowmap);

	waitKey();

	destroyAllWindows();

	return 0;
}

