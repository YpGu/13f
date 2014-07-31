// Latest version in use.
// Must change some parameters manually (e.g. num of topics)

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>
#include <string>
#include <boost/random.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/variate_generator.hpp>

using namespace std;
using namespace boost;

#define NUM_PERSON 1540
#define NUM_BILL 7162
#define NUM_TOPIC 10
#define NUM_WORD 10000
#define NUM_PERSON_BILL	2502466			// num of (non-zero) training person-bill pairs (../data/Person_Bill_train)
#define NUM_PB_TEST 277987			// num of (non-zero) testing person-bill pairs (../data/Person_Bill_test)

double **dmatrix(int row, int col)
{
	double **matrix = (double**)calloc(row, sizeof(double*));
	if (matrix == NULL)
		return NULL;
	for (int i = 0; i < row; i++)
	{
		matrix[i] = (double*)calloc(col, sizeof(double));
		if (matrix[i] == NULL)
			return NULL;
	}
	return matrix;
}

int **dmatrix(int row, int col, bool f)
{
	int **matrix = (int**)calloc(row, sizeof(int*));
	if (matrix == NULL)
		return NULL;
	for (int i = 0; i < row; i++)
	{
		matrix[i] = (int*)calloc(col, sizeof(int));
		if (matrix[i] == NULL)
			return NULL;
	}
	return matrix;
}


void free_dmatrix(double** mat, int row)
{
	for (int i = 0; i < row; i++)
		free(mat[i]);
	free(mat);

	return;
}

void free_imatrix(int **mat, int row)
{
	for (int i = 0; i < row; i++)
		free(mat[i]);
	free(mat);

	return;
}


bool isReal(double x)
{
	return (x==x);
}


double logis(double x)
{
	return exp(x)/(1+exp(x));
}


void normal_dist(double *arr, int len)
{
	boost::mt19937 rng;
	boost::normal_distribution<double> nd(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> > vargen(rng, nd);

//	srand(time(NULL));
//	double r = rand()*100/(double)RAND_MAX;
//	int tmp;
//	for (int i = 0; i < (int)r; i++)
//		tmp = vargen();

	for (int i = 0; i < len; i++)
		arr[i] = vargen();

	return;
}


void normal_dist(double **mat, int row, int col)
{
	boost::mt19937 rng;
	boost::normal_distribution<double> nd(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> > vargen(rng, nd);

//	srand(time(NULL));
//	double r = rand()*100/(double)RAND_MAX;
//	int tmp;
//	for (int i = 0; i < (int)r; i++)
//		tmp = vargen();

	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			mat[i][j] = vargen();

	return;
}


// generate theta from lambda
void lambda2theta(double **lambda_dk, double **theta_dk)
{
	// lambda -> theta
	// using logistic function
	for (int d = 0; d < NUM_BILL; d++)
	{
		double norm_d = 0;
		for (int k = 0; k < NUM_TOPIC - 1; k++)
		{
			if (lambda_dk[d][k] > 100)
			{
				for (int z = 0; z < NUM_TOPIC; z++)
				{
					if (z != k)
					//	theta_dk[d][z] = pow(10,-6);
						theta_dk[d][z] = 0;
					else
					//	theta_dk[d][z] = 1 - (NUM_TOPIC-1)*pow(10,-6);
						theta_dk[d][z] = 1;
				}
//				cout << "lambda too large" << endl;
				goto begin_next_loop;
			}
			theta_dk[d][k] = exp(lambda_dk[d][k]);
			norm_d += theta_dk[d][k];
		}
		theta_dk[d][NUM_TOPIC-1] = 1;
		norm_d += 1;

		// normalize
		if (fabs(norm_d) >= 1)
			for (int k = 0; k < NUM_TOPIC; k++)
				theta_dk[d][k] /= norm_d;
		else
			cout << "Something wrong in lambda2theta." << endl;
begin_next_loop:
		;
	}

	return;
}


void theta2lambda(double **theta_dk, double **lambda_dk)
{
	// theta -> lambda
	// using logistic function
	for (int d = 0; d < NUM_BILL; d++)
	{
		if (theta_dk[d][NUM_TOPIC-1] != 0)
		{
			for (int k = 0; k < NUM_TOPIC-1; k++)
			{
				double q = theta_dk[d][k] / theta_dk[d][NUM_TOPIC-1];
				if (q != 0)
					lambda_dk[d][k] = log(q);
				else
					lambda_dk[d][k] = -10;
			}
		}
		else if (theta_dk[d][NUM_TOPIC-1] != 1)
		{
			theta_dk[d][NUM_TOPIC-1] = pow(10,-9);
			d--;
			continue;
		}
		else
		{
			for (int k = 0; k < NUM_TOPIC-1; k++)
				lambda_dk[d][k] = -10;
		}
	}

	return;
}


// generate p(v_ud = yes) from x, a, b
void generate_prob(double **theta_dk, double **x_uk, double **a_dk, double *b_d, double **p_ud, int **mat, int len)
{
	// p_ud: the return value
	// mat: original training/testing matrix
	// len: number of non-zero terms in mat

	for (int i = 0; i < len; i++)
	{
		int d = mat[i][0];			// d-bill
		int u = mat[i][1];			// u-person
		double vote = (mat[i][2]==1)?1:0;	// vote 1/0

		double ins_logis = 0;
		for (int k = 0; k < NUM_TOPIC; k++)
			ins_logis += theta_dk[d][k] * x_uk[u][k] * a_dk[d][k];
		ins_logis += b_d[d];

		p_ud[u][d] = logis(ins_logis);
	}

	return;
}

// likelihood in the first term (bill * term matrix)
double calcLikelihood(double **lambda_dk, double **beta_wk, int **n_dw, double *p_B, double lambda_B)
{
	double **theta_dk = dmatrix(NUM_BILL, NUM_TOPIC);
	lambda2theta(lambda_dk, theta_dk);

	double likelihood = 0;
	for (int d = 0; d < NUM_BILL; d++)
	{
		for (int i = 0; i < n_dw[d][0]; i++)
		{
			int w = n_dw[d][1+i];
			int freq = n_dw[d][1+i+n_dw[d][0]];

			double sumz = 0;
			for (int z = 0; z < NUM_TOPIC; z++)
				sumz += beta_wk[w][z] * theta_dk[d][z];
			double logV = lambda_B * p_B[w] + (1-lambda_B) * sumz;

			if (logV != 0)
				likelihood += (double)freq * log(logV);
		}
	}

	free_dmatrix(theta_dk, NUM_BILL);

	return likelihood;
}

// likelihood in the second term (bill * person matrix)
double calcLikelihood(double **p_ud, int **mat, int len, bool visible)
{
	// p_ud: prob(v_ud = yes)
	// mat: training/testing matrix
	// len: number of non-zero terms in mat
	// visiable: show results if true

	double likelihood = 0, error = 0;
	double cor = 0;
	int wi_prob = 0;

	for (int i = 0; i < len; i++)
	{
		int d = mat[i][0];			// d-bill
		int u = mat[i][1];			// u-person
		double vote = (mat[i][2]==1)?1:0;	// vote 1/0

		error += fabs(vote-p_ud[u][d]);

		if (vote == 1.0)
		{
			if (p_ud[u][d] < pow(10,-9))
				wi_prob++;
			else
				likelihood += log(p_ud[u][d]);
		}
		else
		{
			if (1-p_ud[u][d] < pow(10,-9))
				wi_prob++;
			else
				likelihood += log(1-p_ud[u][d]);
		}

		if ((vote-0.5) * (p_ud[u][d]-0.5) > 0)
			cor++;
	}

	if (visible)
	{
		cout << "Likelihood = " << likelihood << endl;
		cout << "MAE = " << error/(double)len << endl;
		cout << "Accuracy = " << cor/(double)len << endl;
	}

	return likelihood;
}


// calculate a derivative
void get_derv_1(double **lambda_dk, double **beta_wk, int **n_dw, double *p_B, double **derv_1_dk)
{
	// derv_1_dk is the output

	double **theta_dk = dmatrix(NUM_BILL, NUM_TOPIC);
	lambda2theta(lambda_dk, theta_dk);

	double *p_z_dw = (double*)calloc(NUM_TOPIC, sizeof(double));
	double norm_z_dw;

	// empty this derivative
	for (int d = 0; d < NUM_BILL; d++)
		for (int k = 0; k < NUM_TOPIC; k++)
			derv_1_dk[d][k] = 0;

	// plsa using gradient descent
	for (int d = 0; d < NUM_BILL; d++)
	{
		// calculate p(z|d,w) and the first derivative
		for (int i = 0; i < n_dw[d][0]; i++)
		{
			int w = n_dw[d][1+i];
			int freq = n_dw[d][1+i+n_dw[d][0]];

			norm_z_dw = 0;
			for (int z = 0; z < NUM_TOPIC; z++)
			{
				p_z_dw[z] = beta_wk[w][z] * theta_dk[d][z];	// p(w|z)*p(z|d)
				norm_z_dw += p_z_dw[z];
			}
			if (norm_z_dw != 0)					// sum_{z}{p(w|z)*p(z|d)}
			{
				for (int k = 0; k < NUM_TOPIC; k++)
				{
					p_z_dw[k] /= norm_z_dw;			// p(z|d,w) (normalized)
					double inc = (double)freq * p_z_dw[k];	// n(d,w)*p(z|d,w)

					derv_1_dk[d][k] += inc;
				}
			}
			else
			{
			// do nothing
			//	printf("sum{z}{p(w|z)p(z|d)} = 0!\n");
			}
		}
	}			

/*	// check if derv_1 is too large:
	for (int d = 0; d < NUM_BILL; d++)
	{
		for (int k = 0; k < NUM_TOPIC; k++)
		{
			if (fabs(derv_1_dk[d][k]) > pow(10,6) || !isReal(derv_1_dk[d][k]))
			{
				cout << "line 316: derv_1_dk = " << derv_1_dk[d][k] << endl;
				int error; cin >> error;
			}
		}
	}
*/	
	free_dmatrix(theta_dk, NUM_BILL);
	free(p_z_dw);

	return;
}

// update theta (lambda) by GD
void update_lambda(
	double **lambda_dk, double **beta_wk,				// topic model
	int **n_dw, double *p_B, double lambda_B,			// background model and bill-word mat
	int **mat, int len,						// training/testing matrix and the number of its non-zero elements
	double **derv_1_dk,						// some derivative of the first part
	double **x_uk, double **a_dk, double *b_d,			// voting prediction
	double *pt_eta,							// learning rate
	int iters,							// number of iterations
	double lambda_0,						// linking two likelihoods (0~1)
	double deno_1,							// denominator of the first part
	double deno_2							// denominator of the second part
)
{
	double obj_func[2] = {-1,-1};
	double eta = *pt_eta;
	double init_likelihood;

	// generate theta and p(v_ud = yes)
	double **theta_dk = dmatrix(NUM_BILL, NUM_TOPIC);
	lambda2theta(lambda_dk, theta_dk);				// should be updated once lambda is updated

	double **p_ud = dmatrix(NUM_PERSON, NUM_BILL);
	generate_prob(theta_dk, x_uk, a_dk, b_d, p_ud, mat, len);	// should be updated after theta is updated

	// grad
	double **lambda_inc_dk = dmatrix(NUM_BILL, NUM_TOPIC-1);

	// training
	for (int iter = 0; iter < iters; iter++)
	{
		if (iter%1==0)	cout << "\tIter " << iter << endl;

		cout << "\tStep 1.1" << endl;
		get_derv_1(theta_dk, beta_wk, n_dw, p_B, derv_1_dk);	// get derv_1

		for (int d = 0; d < NUM_BILL; d++)
		{
			for (int l = 0; l < NUM_TOPIC; l++)
			{
				// we alreadly have d(F)/d(theta_dk) here: derv_1
				// here we update \lambda_dk for k in range(NUM_TOPIC-1)
				for (int k = 0; k < NUM_TOPIC-1; k++)	// -1 because there's only K-1 lambda
				{
					if (l != k)			// the first case: l != k
					{
						double grad = (1-lambda_0) * derv_1_dk[d][k] * theta_dk[d][l] / deno_1;
						lambda_inc_dk[d][k] += eta * grad;
					}
					else				// the second case: l < NUM_TOPIC, l == k
					{
						double grad = -(1-lambda_0) * derv_1_dk[d][k] * (1-theta_dk[d][k]) / deno_1;
						lambda_inc_dk[d][k] += eta * grad;
					}
				}
			}
		}

		cout << "\tStep 1.2" << endl;
		for (int ind = 0; ind < NUM_PERSON_BILL; ind++)
		{
			int d = mat[ind][0];				// bill
			int u = mat[ind][1];				// person
			double vote = (mat[ind][2]==1)?1:0;		// vote

			for (int l = 0; l < NUM_TOPIC; l++)
			{
				for (int k = 0; k < NUM_TOPIC - 1; k++)	// -1 because there's only K-1 lambda
				{
					if (l != k)			// the first case: l != k
					{
						double grad = -lambda_0 * x_uk[u][l] * a_dk[d][l] * (vote - p_ud[u][d]) * theta_dk[d][l] * theta_dk[d][k] / deno_2;
//						if (fabs(grad) > pow(10,4))
//						{ cout << "(1)grad = " << grad << endl; int err; cin >> err;}

						lambda_inc_dk[d][k] += eta * grad;
					}
					else				// the second case: l < NUM_TOPIC && l == k
					{
						double grad = lambda_0 * x_uk[u][l] * a_dk[d][l] * (vote - p_ud[u][d]) * theta_dk[d][k] * (1-theta_dk[d][k]) / deno_2;
//						if (fabs(grad) > pow(10,4))
//						{ cout << "(2)grad = " << grad << endl; int err; cin >> err;}

						lambda_inc_dk[d][k] += eta * grad;
					}
				}
			}
		}

//		cout << "::::::::::\nsome lambda_inc_dk:" << endl;
//		for (int k = 0; k < NUM_TOPIC-1; k++)
//			cout << lambda_inc_dk[3235][k] << endl;
//		cout << "::::::::::" << endl;

		// update lambda
		for (int d = 0; d < NUM_BILL; d++)
			for (int k = 0; k < NUM_TOPIC-1; k++)
				lambda_dk[d][k] += lambda_inc_dk[d][k];

	//	free_dmatrix(lambda_inc_dk, NUM_BILL);
		for (int d = 0; d < NUM_BILL; d++)
			for (int k = 0; k < NUM_TOPIC-1; k++)
				lambda_inc_dk[d][k] = 0;

		// check lambda
//		for (int d = 0; d < NUM_BILL; d++)
//			for (int k = 0; k < NUM_TOPIC-1; k++)
//				if (!isReal(lambda_dk[d][k]))
//					{cout << "d = " << d << " k = " << k << endl; int err; cin >> err;}

//		cout << "lambda_dk[3235][2] = " << lambda_dk[3235][2] << endl;
//		for (int k = 0; k < NUM_TOPIC-1; k++)
//			cout << "lambda_dk[3235][" << k << "] = " << lambda_dk[3235][k] << endl;
//		for (int k = 0; k < NUM_TOPIC-1; k++)
//			cout << "lambda_dk[421][" << k << "] = " << lambda_dk[421][k] << endl;

		// update theta and p(v_ud = yes)
		lambda2theta(lambda_dk, theta_dk);
		generate_prob(theta_dk, x_uk, a_dk, b_d, p_ud, mat, len);


		// checking objective function
	//	if (iter%1 == 0)
		if (true)
		{
			generate_prob(theta_dk, x_uk, a_dk, b_d, p_ud, mat, len);

			double likel_1 = calcLikelihood(theta_dk, beta_wk, n_dw, p_B, lambda_B);	// likelihood of the 1st part
			double likel_2 = calcLikelihood(p_ud, mat, NUM_PERSON_BILL, false);		// likelihood of the 2nd part
			likel_1 /= deno_1;								// average likelihood of the 1st part
			likel_2 /= deno_2;								// average likelihood of the 2nd part

	//		cout << "\t1st likelihood = " << likel_1 << endl;
	//		cout << "\t2nd likelihood = " << likel_2 << endl;
			obj_func[1] = (1-lambda_0) * likel_1 + lambda_0 * likel_2;
			if (iter == 0) init_likelihood = obj_func[1];

			double rate = fabs((obj_func[1] - obj_func[0]) / obj_func[0]);
			if (iter%1 == 0)
			{
				cout << "\tOld likelihood (total) = " << obj_func[0] << endl;
				cout << "\tNow likelihood (total) = " << obj_func[1] << endl;
				cout << "\tRate = " << rate << endl;
			}

			if (rate < pow(10,-5) && iter > 1)
			{
				if (obj_func[1] < init_likelihood)
					(*pt_eta) *= 0.9;
				break;
			}
//			if (obj_func[1] < obj_func[0] && iter > 1)
//				break;

			obj_func[0] = obj_func[1];
		}
	}

	free_dmatrix(lambda_inc_dk, NUM_BILL);
	free_dmatrix(theta_dk, NUM_BILL);

	return;
}

// update beta by EM
void update_beta(double **lambda_dk, double **beta_wk, int **n_dw, double *p_B, double lambda_B)
{
	// lambda_B: parameter in the background model

	double **theta_dk = dmatrix(NUM_BILL, NUM_TOPIC);
	lambda2theta(lambda_dk, theta_dk);

	double *new_norm_w_z = (double*)calloc(NUM_TOPIC, sizeof(double));
	double *p_z_dw = (double*)calloc(NUM_TOPIC, sizeof(double));
	double **tmp_pwz = dmatrix(NUM_WORD, NUM_TOPIC);

	for (int d = 0; d < NUM_BILL; d++)
	{
		for (int i = 0; i < n_dw[d][0]; i++)
		{
			int w = n_dw[d][1+i];
			int freq = n_dw[d][1+i+n_dw[d][0]];

			double norm_z_dw = 0;
			for (int z = 0; z < NUM_TOPIC; z++)
			{
				p_z_dw[z] = beta_wk[w][z] * theta_dk[d][z];	// p(w|z)*p(z|d)
				norm_z_dw += p_z_dw[z];
			}
			if (norm_z_dw != 0)					// sum_{z}{p(w|z)*p(z|d)}
			{
				for (int z = 0; z < NUM_TOPIC; z++)
				{
					p_z_dw[z] /= norm_z_dw;			// p(z|d,w) (normalized)
					double inc = (double)freq * p_z_dw[z];	// n(d,w)*p(z|d,w)

					// p(w|z): using background model
					tmp_pwz[w][z] += inc * ((1-lambda_B)*norm_z_dw) / (lambda_B*p_B[w] + (1-lambda_B)*norm_z_dw);
					new_norm_w_z[z] += inc * ((1-lambda_B)*norm_z_dw) / (lambda_B*p_B[w] + (1-lambda_B)*norm_z_dw);

					// p(w|z): not using background model
				//	tmp_pwz[w][z] += inc;			// update p(w|z) (un-normalized)
				//	new_norm_w_z[z] += inc;			// update sum_{w}{p(w|z)}
				}
			}
	//		else printf("sum{z}{p(w|z)p(z|d)} = 0!\n");
		}
	}

	for (int w = 0; w < NUM_WORD; w++)
		for (int z = 0; z < NUM_TOPIC; z++)
			tmp_pwz[w][z] /= new_norm_w_z[z];

	for (int w = 0; w < NUM_WORD; w++)
		for (int z = 0; z < NUM_TOPIC; z++)
			beta_wk[w][z] = tmp_pwz[w][z];

	// checking sum_w{beta_wz} = 1
	for (int k = 0; k < NUM_TOPIC; k++)
	{
		double sum = 0;
		for (int w = 0; w < NUM_WORD; w++)
			sum += beta_wk[w][k];
		if (fabs(sum-1)>pow(10,-9))
		{
			cout << "sum_z{beta_wz} = " << sum << " in the updating part of beta!" << endl;
			int error; cin >> error;
		}
	}

	free_dmatrix(theta_dk, NUM_BILL);
	free_dmatrix(tmp_pwz, NUM_WORD);
	free(new_norm_w_z);
	free(p_z_dw);

	return;
}

// update x_uk, a_dk, b_d
void update_xab(
	double **lambda_dk,						// theta from topic model
	double **x_uk, double **a_dk, double *b_d,			// bill/person popularity/polarity
	int **mat, int len,						// training/testing matrix and the number of its non-zero elements
	double *pt_eta,							// learning rate of GD
	double lambda_0,						// linking two likelihood
	int iters,							// iterations
	bool visible,							// show results if true
	double deno_1,							// denominator of the first part
	double deno_2							// denominator of the second part
)
{
	// mat: training/testing matrix
	// len: number of non-zero elements in mat
	// p_ud: prob(v_ud = yes)
	// eta: learning rate of GD

	double eta = *pt_eta;
	double c = 0.000;						// regularization term
	double init_likelihood;

	// generate theta and p(v_ud = yes)
	double **theta_dk = dmatrix(NUM_BILL, NUM_TOPIC);
	lambda2theta(lambda_dk, theta_dk);				// only once: lambda will not be changed
	double **p_ud = dmatrix(NUM_PERSON, NUM_BILL);
	generate_prob(theta_dk, x_uk, a_dk, b_d, p_ud, mat, len);	// should be updated once x,a,b is updated

	double **x_uk_inc = dmatrix(NUM_PERSON, NUM_TOPIC);
	double **a_dk_inc = dmatrix(NUM_BILL, NUM_TOPIC);
	double *b_d_inc = (double*)calloc(NUM_BILL, sizeof(double));

	double likelihood[2] = {-1,-1};
	for (int iter = 0; iter < iters; iter++)
	{
		cout << "\tIteration " << iter << endl;

		// gradient descent
		for (int ind = 0; ind < NUM_PERSON_BILL; ind++)
		{
			// calculate gradient
			int u = mat[ind][1];			// person
			int d = mat[ind][0];			// bill
			double vote = (mat[ind][2]==1.0)?1:0;	// vote 0/1

			double diff = vote - p_ud[u][d];

			for (int k = 0; k < NUM_TOPIC; k++)
			{
				x_uk_inc[u][k] += lambda_0 * theta_dk[d][k] * a_dk[d][k] * diff / deno_2; 
				a_dk_inc[d][k] += lambda_0 * theta_dk[d][k] * x_uk[u][k] * diff / deno_2;
			}
			b_d_inc[d] += lambda_0 * diff / deno_2;
		}

		// add regularization term
		for (int u = 0; u < NUM_PERSON; u++)
			for (int k = 0; k < NUM_TOPIC; k++)
				x_uk_inc[u][k] -= 2 * c * x_uk[u][k];
		for (int d = 0; d < NUM_BILL; d++)
			for (int k = 0; k < NUM_TOPIC; k++)
				a_dk_inc[d][k] -= 2 * c * a_dk[d][k];

		// update parameters
		for (int u = 0; u < NUM_PERSON; u++)
			for (int k = 0; k < NUM_TOPIC; k++)
				x_uk[u][k] += eta * x_uk_inc[u][k];
		for (int d = 0; d < NUM_BILL; d++)
			for (int k = 0; k < NUM_TOPIC; k++)
				a_dk[d][k] += eta * a_dk_inc[d][k];
		for (int d = 0; d < NUM_BILL; d++)
			b_d[d] += eta * b_d_inc[d];

		// update p(v_ud = yes)
		generate_prob(theta_dk, x_uk, a_dk, b_d, p_ud, mat, len);

		// make gard = 0
		for (int u = 0; u < NUM_PERSON; u++)
			for (int k = 0; k < NUM_TOPIC; k++)
				x_uk_inc[u][k] = 0;
		for (int d = 0; d < NUM_BILL; d++)
			for (int k = 0; k < NUM_TOPIC; k++)
				a_dk_inc[d][k] = 0;
		for (int d = 0; d < NUM_BILL; d++)
			b_d_inc[d] = 0;

		// free inc of x,a,b
	//	free_dmatrix(x_uk_inc, NUM_PERSON);
	//	free_dmatrix(a_dk_inc, NUM_BILL);
	//	free(b_d_inc);


		// checking likelihood
		// note: here we only need to maximize the second likelihood (because x,a,b have no influence on the first one)
		likelihood[1] = calcLikelihood(p_ud, mat, len, visible) / deno_2;
		if (iter == 0)	init_likelihood = likelihood[1];

		double rate = fabs((likelihood[1] - likelihood[0])/ likelihood[0]);
		if (iter%1 == 0)
		{
			cout << "\tOld likelihood (2nd part only) = " << likelihood[0] << endl;
			cout << "\tNew likelihood (2nd part only) = " << likelihood[1] << endl;
			cout << "\tRate = " << rate << endl;
		}

		if (fabs(rate) < pow(10,-5) && iter > 1)
		{
			if (likelihood[1] < init_likelihood)
				(*pt_eta) *= 0.9;
			break;
		}

		likelihood[0] = likelihood[1];
	}

	free_dmatrix(x_uk_inc, NUM_PERSON);
	free_dmatrix(a_dk_inc, NUM_BILL);
	free(b_d_inc);

	free_dmatrix(theta_dk, NUM_BILL);
	free_dmatrix(p_ud, NUM_PERSON);

	return;
}


double rmse_evaluation(int **mat, int len, double **lambda_dk, double **x_uk, double **a_dk, double *b_d)
{
	// mat: training/testing matrix
	// len: number of non-zero elements in mat

	double **theta_dk = dmatrix(NUM_BILL, NUM_TOPIC);
	lambda2theta(lambda_dk, theta_dk);

	double **p_ud = dmatrix(NUM_PERSON, NUM_BILL);
	generate_prob(theta_dk, x_uk, a_dk, b_d, p_ud, mat, len);

//	double cor = 0;		// correct prediction
	double rmse = 0;
	int wi_prob = 0;	// probs that are either too large or too small

	for (int ind = 0; ind < len; ind++)
	{
		int d = mat[ind][0];			// d-bill
		int u = mat[ind][1];			// u-person
		double vote = (mat[ind][2]==1)?1:0;	// vote 1/0

		double diff = vote-p_ud[u][d];
		rmse += diff*diff;

//		if ((vote-0.5) * (p_ud[u][d]-0.5) > 0)
//			cor++;
	}

//	cout << "Accuracy = " << (double)cor/len << endl;
//	cout << "There are " << wi_prob << " probs either too large or too small." << endl;

	free_dmatrix(theta_dk, NUM_BILL);
	free_dmatrix(p_ud, NUM_PERSON);

	rmse /= (double)len;
	rmse = sqrt(rmse);
	return rmse;
}
	


// train the whole model
void train(
	double **lambda_dk, double **beta_wk, 				// theta (lambda) and beta in topic model
	double **x_uk, double **a_dk, double *b_d, 			// voting prediction
	int **n_dw, double *p_B,					// bill-word matrix and bkg model	
	int **trainMat,							// training matrix
	int **testMat,							// testing matrix
	double lambda_0,						// linking two likelihoods
	double lambda_B,						// parameter in background model
	int iters,							// iterations
	double **train_test_likelihood,
	double deno_1,							// denominator of the first part
	double deno_2							// denominator of the second part
)
{
	// objective function = likelihood of topic model + \lambda * likelihood of voting data
	// p(V_ud = yes) = sigma( sigma_k { theta_dk * x_uk * a_dk } + b_d )
	// theta = (some logistic funcion) of lambda_dk

	// Step 1	Update theta_dk (lambda_dk) using stocastic gradient descent
	//	1.1	Calculate the derivate in the first (topic model) part
	//	1.2	Calculate the derivate in the second (voting) part
	// Step 2	Update p(w|z) using EM, take just one step
	// Step 3	Update x_uk, a_dk, b_d using stocastic gradient descent

	// d - bill;   k - topic;   w - word;   u - person(user)


	// parameters in the first part
	double **theta_dk = dmatrix(NUM_BILL, NUM_TOPIC);		// theta[d][k]
	double **derv_1_dk = dmatrix(NUM_BILL, NUM_TOPIC);		// a derivative in the first likelihood

	// parameters in the second part
	double **p_ud = dmatrix(NUM_PERSON, NUM_BILL);			// prob(V_ud = yes)

	// 
	double eta = 1000;			// learning rate
	double *pt_eta = &eta;
	int iterations_for_lambda = 20;		// iterations in step 1
	int iterations_for_xab = 60;		// iterations in step 3

	double obj_func[2] = {-1,-1};		// old/new objective function
	double likel_1, likel_2, rate;

	for (int iter = 0; iter < iters; iter++)
	{
		cout << "Iter " << iter << "/" << iters << endl;

		// preparation
		get_derv_1(lambda_dk, beta_wk, n_dw, p_B, derv_1_dk);

		// train
		cout << "Step 1" << endl;
		update_lambda(lambda_dk, beta_wk, n_dw, p_B, lambda_B, trainMat, NUM_PERSON_BILL,
			derv_1_dk, x_uk, a_dk, b_d, pt_eta, iterations_for_lambda, lambda_0, deno_1, deno_2);
/*
//--- calculate likelihood
//		generate_prob(theta_dk, x_uk, a_dk, b_d, p_ud, trainMat, NUM_PERSON_BILL);

		likel_1 = calcLikelihood(theta_dk, beta_wk, n_dw, p_B, lambda_B);	// likelihood of the 1st part
		likel_2 = calcLikelihood(p_ud, trainMat, NUM_PERSON_BILL, false);	// likelihood of the 2nd part
	//	cout << "---1st likelihood = " << likel_1 << endl;
	//	cout << "---2nd likelihood = " << likel_2 << endl;
		obj_func[1] = likel_1 + lambda_0 * likel_2;
		cout << "---Old likelihood = " << obj_func[0] << endl;
		cout << "---Now likelihood = " << obj_func[1] << endl;

		rate = fabs((obj_func[1] - obj_func[0]) / obj_func[0]);
		cout << "---Rate = " << rate << endl;

		obj_func[0] = obj_func[1];
//--- end of calculating likelihood
*/
		cout << "Step 2" << endl;
		update_beta(lambda_dk, beta_wk, n_dw, p_B, lambda_B);
/*
//--- calculate likelihood
//		generate_prob(theta_dk, x_uk, a_dk, b_d, p_ud, trainMat, NUM_PERSON_BILL);

		likel_1 = calcLikelihood(theta_dk, beta_wk, n_dw, p_B, lambda_B);	// likelihood of the 1st part
		likel_2 = calcLikelihood(p_ud, trainMat, NUM_PERSON_BILL, false);	// likelihood of the 2nd part
	//	cout << "---1st likelihood = " << likel_1 << endl;
	//	cout << "---2nd likelihood = " << likel_2 << endl;
		obj_func[1] = likel_1 + lambda_0 * likel_2;
		cout << "---Old likelihood = " << obj_func[0] << endl;
		cout << "---Now likelihood = " << obj_func[1] << endl;

		rate = fabs((obj_func[1] - obj_func[0]) / obj_func[0]);
		cout << "---Rate = " << rate << endl;


		rate = fabs((obj_func[1] - obj_func[0]) / obj_func[0]);

		obj_func[0] = obj_func[1];
//--- end of calculating likelihood
*/
		cout << "Step 3" << endl;
		update_xab(lambda_dk, x_uk, a_dk, b_d, trainMat, NUM_PERSON_BILL, pt_eta, lambda_0, iterations_for_xab, false, deno_1, deno_2);

		// update theta and p(v_ud = yes) in order to calculate objective function (likelihood)
		lambda2theta(lambda_dk, theta_dk);
		generate_prob(theta_dk, x_uk, a_dk, b_d, p_ud, trainMat, NUM_PERSON_BILL);

		// checking likelihood
		likel_1 = calcLikelihood(theta_dk, beta_wk, n_dw, p_B, lambda_B);
		likel_2 = calcLikelihood(p_ud, trainMat, NUM_PERSON_BILL, false);
		likel_1 /= deno_1;
		likel_2 /= deno_2;
		obj_func[1] = (1-lambda_0) * likel_1 + lambda_0 * likel_2;

		// record training/testing likelihood for every iteration
		train_test_likelihood[0][iter] = obj_func[1];
		train_test_likelihood[1][iter] = (1-lambda_0) * calcLikelihood(theta_dk, beta_wk, n_dw, p_B, lambda_B) / deno_1 
						+ lambda_0 * calcLikelihood(p_ud, testMat, NUM_PB_TEST, false) / deno_2;
		train_test_likelihood[2][iter] = rmse_evaluation(trainMat, NUM_PERSON_BILL, lambda_dk, x_uk, a_dk, b_d);
		train_test_likelihood[3][iter] = rmse_evaluation(testMat, NUM_PB_TEST, lambda_dk, x_uk, a_dk, b_d);

//		cout << "---Old likelihood = " << obj_func[0] << endl;
//		cout << "---Now likelihood = " << obj_func[1] << endl;

		rate = fabs((obj_func[1] - obj_func[0]) / obj_func[0]);
		cout << "---Rate = " << rate << endl;
		if (rate < pow(10,-6))
			break;
//		if (obj_func[1] < obj_func[0])
//			break;

		obj_func[0] = obj_func[1];

		cout << "learning rate = " << *pt_eta << endl;
	}

	return;
}


void evaluation(double **lambda_dk, double **beta_wk, int **n_dw, double *p_B, double lambda_B)
{
	// topic model part: only have training evaluation
	double res = calcLikelihood(lambda_dk, beta_wk, n_dw, p_B, lambda_B);
	cout << "First part likelihood is (only training part) " << res << "\n" << endl;

	return;
}


// evaluate the result, after training is finished
void evaluation(int **mat, int len, double **lambda_dk, double **x_uk, double **a_dk, double *b_d)
{
	// mat: training/testing matrix
	// len: number of non-zero elements in mat

	double **theta_dk = dmatrix(NUM_BILL, NUM_TOPIC);
	lambda2theta(lambda_dk, theta_dk);

	double **p_ud = dmatrix(NUM_PERSON, NUM_BILL);
	generate_prob(theta_dk, x_uk, a_dk, b_d, p_ud, mat, len);


	double error = 0;	// MAE
	double rmse = 0;	// RMSE
	double likelihood = 0;	// likelihood
	double cor = 0;		// correct prediction
	int wi_prob = 0;	// probs that are either too large or too small

	for (int ind = 0; ind < len; ind++)
	{
		int d = mat[ind][0];			// d-bill
		int u = mat[ind][1];			// u-person
		double vote = (mat[ind][2]==1)?1:0;	// vote 1/0

		error += fabs(vote-p_ud[u][d]);
		rmse += (vote-p_ud[u][d])*(vote-p_ud[u][d]);

		if (vote == 1.0)
		{
			if (p_ud[u][d] < pow(10,-9))
				wi_prob++;
			else
				likelihood += log(p_ud[u][d]);
		}
		else
		{
			if (1-p_ud[u][d] < pow(10,-9))
				wi_prob++;
			else
				likelihood += log(1-p_ud[u][d]);
		}

		if ((vote-0.5) * (p_ud[u][d]-0.5) > 0)
			cor++;
	}

	error /= len;
	rmse /= len;
	rmse = sqrt(rmse);

	cout << "MAE = " << error << endl;
	cout << "RMSE = " << rmse << endl;
	cout << "Likelihood = " << likelihood << endl;
	cout << cor << " out of " << len << " correct!" << endl;
	cout << "Accuracy = " << (double)cor/len << endl;
	cout << "There are " << wi_prob << " probs either too large or too small." << endl;

	free_dmatrix(theta_dk, NUM_BILL);
	free_dmatrix(p_ud, NUM_PERSON);

	return;
}

void read_voting_mat(int **trainMat, int **testMat)
{
	FILE *fin;
	fin = fopen("../data/Person_Bill_train", "r");
	int inc = 0;
	while (!feof(fin))
	{
		int billID, personID, vote;
		fscanf(fin, "%d\t%d\t%d\n", &personID, &billID, &vote);
		trainMat[inc][0] = billID;
		trainMat[inc][1] = personID;
		trainMat[inc][2] = vote;
		inc++;
	}
	cout << "inc = " << inc << ", NUM_PERSON_BILL = " << NUM_PERSON_BILL << endl;
	fclose(fin);

	fin = fopen("../data/Person_Bill_test", "r");
	inc = 0;
	while (!feof(fin))
	{
		int billID, personID, vote;
		fscanf(fin, "%d\t%d\t%d\n", &personID, &billID, &vote);
		testMat[inc][0] = billID;
		testMat[inc][1] = personID;
		testMat[inc][2] = vote;
		inc++;
	}
	cout << "inc = " << inc << ", NUM_PB_TEST = " << NUM_PB_TEST << endl;
	fclose(fin);

	return;
}

void read_bill(int **n_dw)
{
	FILE *fin;
	fin = fopen("../data/Bill_Term", "r");
	int *bill_word_len = (int*)calloc(NUM_BILL, sizeof(int));
	while (!feof(fin))
	{
		int billID, wordID, freq;
		fscanf(fin, "%d\t%d\t%d\n", &billID, &wordID, &freq);
		bill_word_len[billID] += 1;
	}
	fclose(fin);

	fin = fopen("../data/Bill_Term", "r");
	for (int d = 0; d < NUM_BILL; d++)
	{
		int billID, wordID, freq;
		int length = bill_word_len[d];
		n_dw[d] = (int*)calloc(2*length+1, sizeof(int));
		n_dw[d][0] = length;
		for (int w = 0; w < length; w++)
		{
			fscanf(fin, "%d\t%d\t%d\n", &billID, &wordID, &freq);
			n_dw[d][1+w] = wordID;
			n_dw[d][1+w+length] = freq;
			if (billID != d)
			{
				cout << "wrong index " << d << endl;
				cout << "billID = " << billID << endl;
				int error; cin >> error;
			}
		}
	}
	fclose(fin);
	free(bill_word_len);

	return;
}

void read_pb(double *p_B)
{
	double sum_pb = 0;
	FILE *fin;
	fin = fopen("../data/DICTWORD_freq", "r");
	while (!feof(fin))
	{
		int wordID, freq;
		fscanf(fin, "%d\t%d\n", &wordID, &freq);
		p_B[wordID] = freq;
		sum_pb += p_B[wordID];
	}
	fclose(fin);
	for (int w = 0; w < NUM_WORD; w++)
		p_B[w] /= sum_pb;

	return;
}


void init(double **lambda_dk, double **beta_wk, double **x_uk, double **a_dk, double *b_d)
{
	normal_dist(lambda_dk, NUM_BILL, NUM_TOPIC - 1);
	normal_dist(x_uk, NUM_PERSON, NUM_TOPIC);
	normal_dist(a_dk, NUM_BILL, NUM_TOPIC);
	normal_dist(b_d, NUM_BILL);

	for (int k = 0; k < NUM_TOPIC; k++)
	{
		double norm = 0;
		for (int w = 0; w < NUM_WORD; w++)
		{
			beta_wk[w][k] = rand() / 100 + 1;
			norm += beta_wk[w][k];
		}
		for (int w = 0; w < NUM_WORD; w++)
			beta_wk[w][k] /= norm;
	}

	return;
}


void writetofile(char *argv[], double **lambda_dk, double **beta_wk, double **x_uk, double **a_dk, double *b_d, double **train_test_record)
{
	// argv[4] is like 'record/XX/10_0_100_'
	double **theta_dk = dmatrix(NUM_BILL, NUM_TOPIC);
	string out_theta = string(argv[4]);
	string out_beta = string(argv[4]);
	string out_x = string(argv[4]);
	string out_a = string(argv[4]);
	string out_b = string(argv[4]);
	string out_tt = string(argv[4]);
	string theta_suffix = "theta";
	string beta_suffix = "beta";
	string x_suffix = "x";
	string a_suffix = "a";
	string b_suffix = "b";
	string tt_suffix = "tt";
	out_theta.append(theta_suffix);
	out_beta.append(beta_suffix);
	out_x.append(x_suffix);
	out_a.append(a_suffix);
	out_b.append(b_suffix);
	out_tt.append(tt_suffix);

	lambda2theta(lambda_dk, theta_dk);

	FILE *fp = fopen(out_theta.c_str(), "w");
	cout << "writing theta to " << out_theta << endl;
	for (int d = 0; d < NUM_BILL; d++)
	{
		for (int k = 0; k < NUM_TOPIC; k++)
			fprintf(fp, "%f\t", theta_dk[d][k]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	fp = fopen(out_beta.c_str(), "w");
	cout << "writing beta to " << out_beta << endl;
	for (int w = 0; w < NUM_WORD; w++)
	{
		for (int k = 0; k < NUM_TOPIC; k++)
			fprintf(fp, "%f\t", beta_wk[w][k]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	fp = fopen(out_x.c_str(), "w");
	cout << "writing x to " << out_x << endl;
	for (int u = 0; u < NUM_PERSON; u++)
	{
		for (int k = 0; k < NUM_TOPIC; k++)
			fprintf(fp, "%f\t", x_uk[u][k]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	fp = fopen(out_a.c_str(), "w");
	cout << "writing a to " << out_a << endl;
	for (int d = 0; d < NUM_BILL; d++)
	{
		for (int k = 0; k < NUM_TOPIC; k++)
			fprintf(fp, "%f\t", a_dk[d][k]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	fp = fopen(out_b.c_str(), "w");
	cout << "writing b to " << out_b << endl;
	for (int d = 0; d < NUM_BILL; d++)
	{
		fprintf(fp, "%f\n", b_d[d]);
	}
	fclose(fp);

	fp = fopen(out_tt.c_str(), "w");
	cout << "writing train/test likelihood to " << out_tt << endl;
	for (int d = 0; d < atoi(argv[3]); d++)
		fprintf(fp, "%f ", train_test_record[0][d]);
	fprintf(fp, "\n");
	for (int d = 0; d < atoi(argv[3]); d++)
		fprintf(fp, "%f ", train_test_record[1][d]);
	fprintf(fp, "\n");
	for (int d = 0; d < atoi(argv[3]); d++)
		fprintf(fp, "%f ", train_test_record[2][d]);
	fprintf(fp, "\n");
	for (int d = 0; d < atoi(argv[3]); d++)
		fprintf(fp, "%f ", train_test_record[3][d]);
	fprintf(fp, "\n");
	fclose(fp);
}


int main(int argc, char *argv[])
{
//	srand((unsigned)time(NULL));

	if (argc < 5)
	{
		cout << "Usage: ./a.out lambda_0 lambda_B iter outputdir_prefix" << endl;
		cout << "Example:" << endl;
		cout << "./logis.out 600 0.6 100 record/10topic/600_0.6_100_" << endl;
		return 1;
	}

	double **lambda_dk = dmatrix(NUM_BILL, NUM_TOPIC - 1);
	double **beta_wk = dmatrix(NUM_WORD, NUM_TOPIC);
	double **x_uk = dmatrix(NUM_PERSON, NUM_TOPIC);
	double **a_dk = dmatrix(NUM_BILL, NUM_TOPIC);
	double *b_d = (double*)calloc(NUM_BILL, sizeof(double));

	int **trainMat = dmatrix(NUM_PERSON_BILL, 3, true);
	int **testMat = dmatrix(NUM_PB_TEST, 3, true);
	double **train_test_record = dmatrix(4, atoi(argv[3]));

	int **n_dw = (int**)calloc(NUM_BILL, sizeof(int*));
	double *p_B = (double*)calloc(NUM_WORD, sizeof(double));


	// Read Training & Testing Dataset
	read_voting_mat(trainMat, testMat);

	// Read Bills
	read_bill(n_dw);

	// Read p_B in bkg
	read_pb(p_B);

	// Calculate two denominators
	double deno_1 = 0, deno_2 = NUM_PERSON_BILL;
	for (int d = 0; d < NUM_BILL; d++)
	{
		int length = n_dw[d][0];
		for (int w = 0; w < length; w++)
		{
			int wordID = n_dw[d][1+w];
			int freq = n_dw[d][1+w+length];
			deno_1 += freq;
		}
	}


	/// Initializing theta (lambda), beta, x_uk, a_dk, b_d
	init(lambda_dk, beta_wk, x_uk, a_dk, b_d);


	/// Training and testing
	cout << "-------Training starts-------" << endl;
	train(lambda_dk, beta_wk, x_uk, a_dk, b_d, n_dw, p_B, trainMat, testMat, atof(argv[1]), atof(argv[2]), atoi(argv[3]), train_test_record, deno_1, deno_2);


	cout << "-------Training Results-------" << endl;
	evaluation(lambda_dk, beta_wk, n_dw, p_B, 0);
	evaluation(trainMat, NUM_PERSON_BILL, lambda_dk, x_uk, a_dk, b_d);
	cout << "------------------------------" << endl;


	cout << "-------Testing Results--------" << endl;
	evaluation(testMat, NUM_PB_TEST, lambda_dk, x_uk, a_dk, b_d);
	cout << "------------------------------" << endl;

	writetofile(argv, lambda_dk, beta_wk, x_uk, a_dk, b_d, train_test_record);


	// Free memory
	free_dmatrix(x_uk, NUM_PERSON);
	free_dmatrix(a_dk, NUM_BILL);
	free_dmatrix(lambda_dk, NUM_BILL);
	free_dmatrix(beta_wk, NUM_WORD);
	free_dmatrix(train_test_record, 4);
	free(b_d);

	free_imatrix(trainMat, NUM_PERSON_BILL);
	free_imatrix(testMat, NUM_PB_TEST);
	free_imatrix(n_dw, NUM_BILL);

	free(p_B);

	return 1;
}

