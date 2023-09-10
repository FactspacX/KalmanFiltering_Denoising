#pragma once
#include <opencv.hpp>
#include <iostream>
#include <string>
#include <time.h>
#include "Calculate_Affine_Matrix.h"

using namespace std;
using namespace cv;

class KFDenoising
{
public:
	KFDenoising(int h, int w, bool s_d, bool t_d);
	Mat run(Mat img_updated, float T_a, float T_b);
	double cost_time;
	vector<float> var_list;

	Mat mask_predict;
	Mat mask_observed;

private:
	// spatial denoising
	Mat spatial_denoising(Mat img);
	void DetectZero(vector<Mat>& input, float threshold);
	float cal_sim(Mat b1, Mat b2);

	void remove_distortion_features(vector<Vec2i>& features, vector<float> controller_parameters);
	Vec2i remove_distortion_tip(Vec2i tip, float x, float y);
	Mat remove_distortion_img(Mat img_distorted, float x, float y);
	Mat Translate_img(Mat img_prev, Mat T);

	void update_distortion_information();

	void Kalman_Motion_Translation(Mat Trans, Mat Trans_f, Mat& T, float a1, float a2);

	void Kalman_Img_update(Mat img_predict, Mat img_observed);
	Mat Kalman_Img(Mat img_predict, Mat img_observed);

	void initialize_mat_motion();

	Vec2i find_tip(Mat img);

	// judge the denoising mode
	bool start_dynamic;
	bool start;
	bool stay;

	bool tip_down;

	// image basic information
	int height;
	int width;

	int count;

	// stored image
	Mat img_prev_denoised;
	Mat img_prev;


	// stored prev features
	vector<Vec2i> features_distorted_prev;
	vector<Vec2i> features_distorted_now;
	vector<Vec2i> features_prev;
	vector<Vec2i> features_now;

	// stored information for motion distortion estimate
	Vec2i prev_distorted_tip;
	Vec2i now_distorted_tip;
	Vec2i prev_tip;
	Vec2i now_tip;

	vector<float> p_img;
	vector<float> q_img;
	vector<float> r_img;
	vector<float> k_img;

	Mat P_motion;
	Mat Q_motion;
	Mat R_motion;
	Mat K_motion;
	Mat I_motion;

	Mat Q_Translation_Motion;
	Mat R_Translation_Motion;
	Mat K_Translation_Motion;
};

