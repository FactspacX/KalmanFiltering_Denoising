#pragma once
#include <iostream>
#include <math.h>
#include <opencv.hpp>
#include <time.h>

using namespace cv;

Mat invert_TR(Mat TR);
Mat trans_matrix(int x, int y, float a, float x0, float b, float y0, float c, float z0, float theta, float t);
Mat Trans_Matrix(float a, float x0, float b, float y0, float c, float z0, float theta, float t, float T_a, float T_b);
Mat Translation_Affine_Matrix(float t, float T_a, float T_b);
