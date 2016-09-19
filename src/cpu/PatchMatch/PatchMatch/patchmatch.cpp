#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
using namespace std;
using namespace cv;


int patch_w = 7;
int	pm_iters = 5;
int rs_max = INT_MAX;

#define XY_TO_INT(x, y) (((y)<<12)|(x))
#define INT_TO_X(v) ((v)&((1<<12)-1))
#define INT_TO_Y(v) ((v)>>12)

//manhattan distance between two patches
int dist(Mat a, Mat b, int ax, int ay, int bx, int by, int cutoff=INT_MAX){
	int ans = 0;
	for (int dy = 0; dy < patch_w; dy++) {	
		for (int dx = 0; dx < patch_w; dx++) {
			Vec3b ai = a.at<Vec3b>(ay + dy, ax + dx);
			Vec3b bi = b.at<Vec3b>(by + dy, bx + dx);
			int dr = abs(ai.val[2] - bi.val[2]);
			int dg = abs(ai.val[1] - bi.val[1]);
			int db = abs(ai.val[0] - bi.val[0]);
			ans += dr*dr + dg*dg + db*db;
		}
		if (ans >= cutoff) { return cutoff; }
	}
	return ans;
}


void improve_guess(Mat a, Mat b, int ax, int ay, int &xbest, int &ybest, int &dbest, int bx, int by) {
	int d = dist(a, b, ax, ay, bx, by, dbest);
	if (d < dbest) {
		dbest = d;
		xbest = bx;
		ybest = by;
	}
}

//get the approximate nearest neighbor and set it into ann
void patchmatch(Mat a, Mat b, int **&ann, int **&annd) {
	/* Initialize with random nearest neighbor field (NNF). */
	ann = new int *[a.rows];
	annd = new int *[a.rows];
	for (int i = 0; i < a.rows; i++)
	{
		ann[i] = new int[a.cols];
		annd[i] = new int[a.cols];
		memset(ann[i], 0, a.cols);
		memset(annd[i], 0, a.cols);
	}
	int aew = a.cols - patch_w + 1, aeh = b.rows - patch_w + 1;       /* Effective width and height (possible upper left corners of patches). */
	int bew = b.cols - patch_w + 1, beh = b.rows - patch_w + 1;

	for (int ay = 0; ay < aeh; ay++) {
		for (int ax = 0; ax < aew; ax++) {
			int bx = rand() % bew;
			int by = rand() % beh;
			ann[ay][ax] = XY_TO_INT(bx, by);
			annd[ay][ax] = dist(a, b, ax, ay, bx, by);
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
				int v = ann[ay][ax];
				int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
				int dbest = annd[ay][ax];

				/* Propagation: Improve current guess by trying instead correspondences from left and above (below and right on odd iterations). */
				if ((unsigned)(ax - xchange) < (unsigned)aew && (ax-xchange)>=0) {
					int vp = ann[ay][ax - xchange];
					int xp = INT_TO_X(vp) + xchange, yp = INT_TO_Y(vp);
					if ((unsigned)xp < (unsigned)bew) {
						improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
					}
				}

				if ((unsigned)(ay - ychange) < (unsigned)aeh && (ay - ychange) >= 0) {
					int vp = ann[ay - ychange][ax];
					int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + ychange;
					if ((unsigned)yp < (unsigned)beh) {
						improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
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
					improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
				}

				ann[ay][ax] = XY_TO_INT(xbest, ybest);
				annd[ay][ax] = dbest;
			}
		}
	}
}

Mat reconstruct(Mat a, Mat b, int ** ann){
	Mat a_recon;
	a.copyTo(a_recon);

	int aew = a.cols - patch_w + 1, aeh = b.rows - patch_w + 1;       /* Effective width and height (possible upper left corners of patches). */
	int bew = b.cols - patch_w + 1, beh = b.rows - patch_w + 1;
	int ystart = 0, yend = aeh, ychange = 1;
	int xstart = 0, xend = aew, xchange = 1;
	int ybest = 0, xbest = 0, v=0;
	for (int ay = ystart; ay != yend; ay += ychange) {
		for (int ax = xstart; ax != xend; ax += xchange) {
			v = ann[ay][ax];
			xbest = INT_TO_X(v);
			ybest = INT_TO_Y(v);

			for (int dy = 0; dy < patch_w; dy++) {
				for (int dx = 0; dx < patch_w; dx++) {
					//Vec3b ai = a_recon.at<Vec3b>(ay + dy, ax + dx);
					Vec3b bi = b.at<Vec3b>(ybest + dy, xbest + dx);
					a_recon.at<Vec3b>(ay + dy, ax + dx).val[2] = bi.val[2];
					a_recon.at<Vec3b>(ay + dy, ax + dx).val[1] = bi.val[1];
					a_recon.at<Vec3b>(ay + dy, ax + dx).val[0] = bi.val[0];
					
				}
				
			}
		}
	}
	return a_recon;
}

int main(int argc, char **argv)
{

	String window_name = "reconstructed";

	// define img matrix
	Mat a = imread("Image/disp5.png");
	Mat b = imread("Image/view5.png");
	Mat a_recon;

	// define and initialize ann and annd array
	int **annd , **ann;	
	patchmatch(a, b, ann, annd);

	a_recon = reconstruct(a, b, ann);

	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	imshow(window_name, a_recon);

	namedWindow("a", CV_WINDOW_AUTOSIZE);
	imshow("a", a);

	namedWindow("b", CV_WINDOW_AUTOSIZE);
	imshow("b", b);

	waitKey();

	destroyAllWindows();

	return 0;
}