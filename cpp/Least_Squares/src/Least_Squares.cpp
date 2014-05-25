#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SVD>

using namespace std;
using namespace Eigen;

VectorXi randperm(int n)
{
	VectorXi sequence = VectorXi::LinSpaced(n, 0, n - 1);
	for(int i = 0; i < n; i++) {
		int index = rand() % (n - i);
		int tmp = sequence.coeff(n - i - 1);
		sequence.coeffRef(n - i - 1) = sequence.coeff(index);
		sequence.coeffRef(index) = tmp;
	}
	return sequence;
}

VectorXd logspace(int n, double start, double end)
{
	VectorXd logspace_vec = VectorXd::LinSpaced(n, start, end);
	for(int i = 0; i < n; i++) {
		logspace_vec.coeffRef(i) = pow(10.0, logspace_vec.coeff(i));
	}
	return logspace_vec;
}

MatrixXd getCenterMat(const MatrixXd &x, int b)
{
	int n = x.rows(), dim = x.cols();
	MatrixXd center(b, dim);
	VectorXi center_index = randperm(n).segment(0, b);
	for(int i = 0; i < b; i++) {
		center.row(i) = x.row(center_index.coeff(i));
	}
	return center;
}

MatrixXd getDesignMat(const MatrixXd &x, const MatrixXd &center, double sigma)
{
	int n = x.rows();
	int b = center.rows();
	MatrixXd K(n, b);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < b; j++) {
			double xx = (x.row(i) - center.row(j)).squaredNorm();
			K.coeffRef(i, j) = exp(-xx/(2*sigma*sigma));
		}
	}
	
	return K;
}

MatrixXd CrossValidationL2LS(const MatrixXd &dist2, const MatrixXd &y, 
	const VectorXd &sigma_vec, const VectorXd &lambda_vec, int b, int fold)
{
	int n = y.rows();
	int n_sigma = sigma_vec.rows(), n_lambda = lambda_vec.rows();
	MatrixXd scores = MatrixXd::Zero(n_sigma, n_lambda);
	VectorXi fold_index = randperm(n) * fold / n;
	
	int *n_cv = new int[fold];
	vector<VectorXd> y_cv;
	vector<VectorXd> y_te;
	for(int fold_ind = 0; fold_ind < fold; fold_ind++) {
		n_cv[fold_ind] = (fold_index.array() == fold_ind).count();
		VectorXd tmp_cv(n - n_cv[fold_ind]);
		VectorXd tmp_te(n_cv[fold_ind]);
		for(int i = 0, j = 0, k = 0; i < n; i++) {
			if(fold_index.coeff(i) == fold_ind) {
				tmp_te.row(j++)=y.row(i);
			}
			else {
				tmp_cv.row(k++)=y.row(i);
			}
		}
		y_cv.push_back(tmp_cv);
		y_te.push_back(tmp_te);
	}
	
	for(int sigma_ind = 0; sigma_ind < n_sigma; sigma_ind++) {
		vector<MatrixXd> K_cv;
		vector<MatrixXd> K_te;
		double t_sigma = sigma_vec.coeff(sigma_ind);
		MatrixXd K = exp(-dist2.array() / (2 * t_sigma * t_sigma));
		for(int fold_ind = 0; fold_ind < fold; fold_ind++) {
			MatrixXd tmp_cv(n - n_cv[fold_ind], b);
			MatrixXd tmp_te(n_cv[fold_ind], b);
			for(int i = 0, j = 0, k = 0; i < n; i++) {
				if(fold_index.coeff(i) == fold_ind) {
					tmp_te.row(j++)=K.row(i);
				}
				else {
					tmp_cv.row(k++)=K.row(i);
				}
			}
			K_cv.push_back(tmp_cv);
			K_te.push_back(tmp_te);
		}	
		for(int lambda_ind = 0; lambda_ind < n_lambda; lambda_ind++) {
			double t_lambda = lambda_vec.coeff(lambda_ind);
			for(int fold_ind = 0; fold_ind < fold; fold_ind++) {
				VectorXd alpha_cv = (K_cv[fold_ind].transpose() * K_cv[fold_ind] + t_lambda * MatrixXd::Identity(b, b))
					.ldlt().solve(K_cv[fold_ind].transpose() * y_cv[fold_ind]);
				scores(sigma_ind, lambda_ind) += (K_te[fold_ind] * alpha_cv - y_te[fold_ind]).squaredNorm() / fold;
			}
		}
	}
	delete [] n_cv;
	
	return scores;
}

void demo_L2LS()
{
	int n = 100, b = 20, fold = 5, dim = 1;
	double x_start = 0, x_end = 2 * M_PI, noise_level = 0.1;
	VectorXd sigma_vec = logspace(5, -2, 2);
	VectorXd lambda_vec = logspace(5, -2, 1);
	
	VectorXd x = VectorXd::LinSpaced(n, x_start, x_end);
	VectorXd y = sin(x.array()).matrix() + noise_level * VectorXd::Random(n);
	
	VectorXi center_index = randperm(n).head(b);
	MatrixXd center_x(dim, b);
	for(int i = 0; i < b; i++) {
		center_x.col(i).noalias() = x.row(center_index.coeff(i));
	}
	MatrixXd dist2 = (x.rowwise().squaredNorm().replicate(1, b) - 2 * x * center_x
		+ center_x.colwise().squaredNorm().replicate(n, 1)).eval();
	MatrixXd scores = CrossValidationL2LS(dist2, y, sigma_vec, lambda_vec, b, fold);
	MatrixXd::Index minRow, minCol;
	double minValue = scores.minCoeff(&minRow, &minCol);
	double sigma = sigma_vec.coeff(minRow), lambda = lambda_vec.coeff(minCol);
	
	MatrixXd K = exp(-dist2.array() / (2 * sigma * sigma));
	VectorXd alpha = (K.transpose() * K + lambda * MatrixXd::Identity(b, b))
		.ldlt().solve(K.transpose() * y);
	
	VectorXd x_te = VectorXd::LinSpaced(200, x_start, x_end);
	VectorXd y_te = K * alpha;
	
//	MatrixXd ret(dim, b + 1);
//	ret.col(0).noalias() = alpha;
//	ret.block(0, 1, dim, b).noalias() = center_x;

//	figure("Least Squares");
//	plot(x, y, "ro");
//	plot(x_te, y_te, "b-");
//	show();
}

