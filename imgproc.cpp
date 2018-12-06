#include "imgproc.h"

namespace IPCVL {
	namespace IMG_PROC {
		void calcHist(cv::InputArray src, int* histogram) {
			cv::Mat inputMat = src.getMat();

			for (int y = 0; y < inputMat.rows; y++) {
				for (int x = 0; x < inputMat.cols; x++) {
					// Todo : histogram을 쌓습니다. 

					/** your code here! **/
					histogram[inputMat.at<uchar>(y, x)]++;
					// hint 1 : for loop 를 이용해서 cv::Mat 순회 시 (1채널의 경우) 
					// inputMat.at<uchar>(y, x)와 같이 데이터에 접근할 수 있습니다. 
				}
			}
		}
		void backprojectHistogram(cv::InputArray src_hsv, cv::InputArray face_hsv, cv::OutputArray dst) {
			cv::Mat srcMat = src_hsv.getMat();
			cv::Mat faceMat = face_hsv.getMat();
			dst.create(srcMat.size(), CV_64FC1);
			std::vector<cv::Mat> channels;

			split(src_hsv, channels);
			cv::Mat mat_h = channels[0];
			cv::Mat mat_s = channels[1];
			cv::Mat outputProb = dst.getMat();

			outputProb.setTo(cv::Scalar(0.));

			double model_hist[64][64] = { { 0., } };
			double input_hist[64][64] = { { 0., } };

			// Todo : hs 2차원 히스토그램을 계산하는 함수를 작성합니다. 
			calcHist_hs(srcMat, input_hist);
			calcHist_hs(faceMat, model_hist);

			for (int y = 0; y < srcMat.rows; y++) {
				for (int x = 0; x < srcMat.cols; x++) {
					// Todo : 양자화된 h,s 값을 얻고 histogram에 값을 더합니다. 
					/** your code here! **/
					outputProb.at<double>(y, x) = UTIL::h_r(model_hist, input_hist, UTIL::quantize(mat_h.at<uchar>(y, x)), UTIL::quantize(mat_s.at<uchar>(y, x)));

					// hint 1 : UTIL::quantize()를 이용해서 srtMat의 값을 양자화합니다. 
					// hint 2 : UTIL::h_r() 함수를 이용해서 outputPorb 값을 계산합니다. 
				}
			}
		}
		void calcHist_hs(cv::InputArray src_hsv, double histogram[][64]) {
			cv::Mat hsv = src_hsv.getMat();
			std::vector<cv::Mat> channels;
			split(hsv, channels);
			cv::Mat mat_h = channels[0];
			cv::Mat mat_s = channels[1];

			// 2차원 히스토그램을 쌓습니다. 
			for (int y = 0; y < hsv.rows; y++) {
				for (int x = 0; x < hsv.cols; x++) {
					// Todo : 양자화된 h,s 값을 얻고 histogram에 값을 더합니다. 
					histogram[UTIL::quantize(mat_h.at<uchar>(y, x))][UTIL::quantize(mat_s.at<uchar>(y, x))]++;
					/** your code here! **/

					// hint 1 : 양자화 시 UTIL::quantize() 함수를 이용해서 mat_h, mat_s의 값을 양자화시킵니다. 
				}
			}

			// 히스토그램을 (hsv.rows * hsv.cols)으로 정규화합니다. 
			for (int j = 0; j < 64; j++) {
				for (int i = 0; i < 64; i++) {
					// Todo : histogram에 있는 값들을 순회하며 (hsv.rows * hsv.cols)으로 정규화합니다. 
					histogram[j][i] /= (hsv.rows * hsv.cols);
					/** your code here! **/
				}
			}
		}
		void thresh_binary(cv::InputArray src, cv::OutputArray dst, const int& threshold) {
			cv::Mat input = src.getMat();
			dst.create(input.size(), CV_8UC1);
			cv::Mat output = dst.getMat();
			for (int y = 0; y < src.rows(); y++) {
				for (int x = 0; x < src.cols(); x++) {
					if (input.at<uchar>(y, x) > threshold) {
						output.at<uchar>(y, x) = 255;
					}
					else {
						output.at<uchar>(y, x) = 0;
					}
				}
			}
		}
		void thresh_otsu(cv::InputArray src, cv::OutputArray dst) {
			cv::Mat input = src.getMat();
			dst.create(input.size(), CV_8UC1);
			cv::Mat output = dst.getMat();

			int range = 256;

			int* hist = new int[range];
			double* hist_hat = new double[range];
			double* v_score = new double[range];

			for (int i = 0; i < range; i++) {
				hist[i] = 0;
				hist_hat[i] = 0;
				v_score[i] = 0;
			}

			// calc hist
			for (int y = 0; y < src.rows(); y++) {
				for (int x = 0; x < src.cols(); x++) {
					hist[input.at<uchar>(y, x)]++;
				}
			}
			// calc hist_hat
			for (int i = 0; i < range; i++) {
				hist_hat[i] = ((double)hist[i] / (src.rows() * src.cols()));
			}

			for (int i = 0; i < range; i++) {
				double w0 = 0;
				double w1 = 0;

				// calc w0
				for (int j = 0; j <= i; j++) {
					w0 += hist_hat[j];
				}
				// calc w1
				for (int k = i + 1; k < range; k++) {
					w1 += hist_hat[k];
				}

				double u0 = 0;
				double u1 = 0;
				// calc u0
				for (int j = 0; j <= i; j++) {
					u0 += j * hist_hat[j];
				}
				// exception for denominator
				if (w0 != 0) {
					u0 /= w0;
				}
				// calc u1
				for (int k = i + 1; k < range; k++) {
					u1 += k * hist_hat[k];
				}
				// exception for denominator
				if (w1 != 0) {
					u1 /= w1;
				}
				double v0 = 0;
				double v1 = 0;

				// calc v0
				for (int j = 0; j <= i; j++) {
					v0 += hist_hat[j] * std::pow((j - u0), 2);
				}
				// exception for denominator
				if (w0 != 0) {
					v0 /= w0;
				}

				// calc v1
				for (int k = i + 1; k < range; k++) {
					v1 += hist_hat[k] * std::pow((k - u1), 2);
				}

				// exception for denominator
				if (w1 != 0) {
					v1 /= w1;
				}
				v_score[i] = w0*v0 + w1*v1;
			}

			double min = std::numeric_limits<double>::max();
			int index = 0;

			// find T
			for (int i = 0; i < range; i++) {
				if (v_score[i] < min) {
					min = v_score[i];
					index = i;
				}
			}
			for (int y = 0; y < src.rows(); y++) {
				for (int x = 0; x < src.cols(); x++) {
					if (input.at<uchar>(y, x) > index) {
						output.at<uchar>(y, x) = 255;
					}
					else {
						output.at<uchar>(y, x) = 0;
					}
				}
			}
		}
		void flood_fill4(cv::Mat & l, const int & j, const int & i, const int & label)
		{
			if (l.at<int>(j, i) == -1) {
				l.at<int>(j, i) = label;
				flood_fill4(l, j, i + 1, label);
				flood_fill4(l, j - 1, i, label);
				flood_fill4(l, j, i - 1, label);
				flood_fill4(l, j + 1, i, label);
			}
		}
		void flood_fill8(cv::Mat & l, const int & j, const int & i, const int & label)
		{
			if (l.at<int>(j, i) == -1) {
				l.at<int>(j, i) = label;
				flood_fill8(l, j - 1, i - 1, label);
				flood_fill8(l, j - 1, i, label);
				flood_fill8(l, j - 1, i + 1, label);
				flood_fill8(l, j, i - 1, label);
				flood_fill8(l, j, i + 1, label);
				flood_fill8(l, j + 1, i - 1, label);
				flood_fill8(l, j + 1, i, label);
				flood_fill8(l, j + 1, i + 1, label);
			}
		}
		void efficient_flood_fill4(cv::Mat & l, const int & j, const int & i, const int & label)
		{
			std::pair<int, int> p = std::make_pair(j, i);
			std::queue <std::pair<int, int>> q;
			
			q.push(p);

			while (!q.empty()) {
				std::pair<int, int> point = q.front();
				q.pop();

				int y = point.first;
				int x = point.second;
				if (l.at<int>(y, x) == -1) {
					int left = x;
					int right = x;

					while (l.at<int>(y, left - 1) == -1) {
						if (left <= 1)
							break;
						left--;
					}

					while (l.at<int>(y, right + 1) == -1) {
						if (right >= l.cols - 1)
							break;
						right++;
					}

					for (int k = left; k <= right; k++) {
						l.at<int>(y, k) = label;
						if (l.at<int>(y - 1, k) == -1 && (k == left || l.at<int>(y - 1, k - 1) != -1)) q.push(std::pair<int, int>(y - 1, k));
						if (l.at<int>(y + 1, k) == -1 && (k == left || l.at<int>(y + 1, k - 1) != -1)) q.push(std::pair<int, int>(y + 1, k));
					}
				}
			}
		}

		void flood_fill(cv::InputArray src, cv::OutputArray dst, const UTIL::CONNECTIVITIES & direction)
		{
			int label = 1;

			cv::Mat input = src.getMat();
			dst.create(src.size(), CV_32SC1);
			cv::Mat output = dst.getMat();
			output.setTo(0);

			for (int y = 0; y < src.rows(); y++) {
				for (int x = 0; x < src.cols(); x++) {
					if (input.at<uchar>(y, x) != 0)
						output.at<int>(y, x) = -1;
					if (y == 0 || x == 0 || y == src.rows() - 1 || x == src.cols() - 1)
						output.at<int>(y, x) = 0;
				}
			}

			if (direction == 0) {
				for (int y = 1; y < src.rows() - 1; y++) {
					for (int x = 1; x < src.cols() - 1; x++) {
						if (output.at<int>(y, x) == -1) {
							flood_fill4(output, y, x, label);
							label++;
						}
					}
				}
			}
			else if (direction == 1) {
				for (int y = 1; y < src.rows() - 1; y++) {
					for (int x = 1; x < src.cols() - 1; x++) {
						if (output.at<int>(y, x) == -1) {
							flood_fill8(output, y, x, label);
							label++;
						}
					}
				}
			}
			else if (direction == 2) {
				for (int y = 1; y < src.rows() - 1; y++) {
					for (int x = 1; x < src.cols() - 1; x++) {
						if (output.at<int>(y, x) == -1) {
							efficient_flood_fill4(output, y, x, label);
							label++;
						}
					}
				}
			}
		}
	}  // namespace IMG_PROC
}

