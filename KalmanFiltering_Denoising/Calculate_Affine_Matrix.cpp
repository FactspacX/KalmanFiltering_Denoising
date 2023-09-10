#include "Calculate_Affine_Matrix.h"

Mat trans_matrix(int x, int y, float a, float x0, float b, float y0, float c, float z0, float theta, float t) {
    float theta_t = theta * t;
    float k = (a * x + b * y - a * x0 - b * y0 - c * z0) / (a * a + b * b + c * c);
    float xr = k * a + x0;
    float yr = k * b + y0;
    float zr = k * c + z0;
    float nx01 = pow((a * a + b * b + c * c), 0.5);
    float x01_x = a / nx01;
    float x01_y = b / nx01;
    float x01_z = c / nx01;

    float y01_x_ = xr - x;
    float y01_y_ = yr - y;
    float y01_z_ = zr - 0;

    float ny01 = pow((a * a + b * b), 0.5);
    float y01_x, y01_y, y01_z;
    if (ny01 > 0) {
        y01_x = b / ny01;
        y01_y = -a / ny01;
        y01_z = 0;
    }
    else {
        y01_x = 1.0f;
        y01_y = 0;
        y01_z = 0;
    }

    float z01_x = x01_y * y01_z - x01_z * y01_y;
    float z01_y = x01_z * y01_x - x01_x * y01_z;
    float z01_z = x01_x * y01_y - x01_y * y01_x;

    Mat TR = Mat::zeros(Size(4, 4), CV_32FC1);
    float* ptr_TR = (float*)TR.data;
    ptr_TR[0] = x01_x;
    ptr_TR[4] = x01_y;
    ptr_TR[8] = x01_z;
    ptr_TR[1] = y01_x;
    ptr_TR[5] = y01_y;
    ptr_TR[9] = y01_z;
    ptr_TR[2] = z01_x;
    ptr_TR[6] = z01_y;
    ptr_TR[10] = z01_z;
    ptr_TR[3] = y01_x_;
    ptr_TR[7] = y01_y_;
    ptr_TR[11] = y01_z_;
    ptr_TR[15] = 1.0f;

    Mat Rx = Mat::zeros(Size(4, 4), CV_32FC1);
    float* ptr_Rx = (float*)Rx.data;
    ptr_Rx[0] = 1.0f;
    ptr_Rx[15] = 1.0f;
    ptr_Rx[5] = cos(theta_t);
    ptr_Rx[6] = -sin(theta_t);
    ptr_Rx[9] = sin(theta_t);
    ptr_Rx[10] = cos(theta_t);

    Mat TR_I = invert_TR(TR);
    Mat Trans = TR * Rx * TR_I;
    return Trans;
}

Mat invert_TR(Mat TR) {
    Mat TR_I = Mat::zeros(Size(4, 4), CV_32FC1);
    float* ptr_TR = (float*)TR.data;
    float* ptr_TR_I = (float*)TR_I.data;

    ptr_TR_I[0] = ptr_TR[0];
    ptr_TR_I[1] = ptr_TR[4];
    ptr_TR_I[2] = ptr_TR[8];
    ptr_TR_I[4] = ptr_TR[1];
    ptr_TR_I[5] = ptr_TR[5];
    ptr_TR_I[6] = ptr_TR[9];
    ptr_TR_I[8] = ptr_TR[2];
    ptr_TR_I[9] = ptr_TR[6];
    ptr_TR_I[10] = ptr_TR[10];

    ptr_TR_I[3] = -ptr_TR_I[0] * ptr_TR[3] - ptr_TR_I[1] * ptr_TR[7] - ptr_TR_I[2] * ptr_TR[11];
    ptr_TR_I[7] = -ptr_TR_I[4] * ptr_TR[3] - ptr_TR_I[5] * ptr_TR[7] - ptr_TR_I[6] * ptr_TR[11];
    ptr_TR_I[11] = -ptr_TR_I[8] * ptr_TR[3] - ptr_TR_I[9] * ptr_TR[7] - ptr_TR_I[10] * ptr_TR[11];

    ptr_TR_I[15] = 1.0f;

    return TR_I;
}

Mat Trans_Matrix(float a, float x0, float b, float y0, float c, float z0, float theta, float t, float T_a, float T_b) {
    Mat T1 = trans_matrix(0, 0, a, x0, b, y0, c, z0, theta, t);
    Mat Trans(3, 3, CV_32FC1);
    float* ptr_T1 = (float*)T1.data;
    float* ptr_Trans = (float*)Trans.data;

    ptr_Trans[0] = ptr_T1[0];
    ptr_Trans[1] = ptr_T1[1];
    ptr_Trans[2] = ptr_T1[3] + T_a * t;
    ptr_Trans[3] = ptr_T1[4];
    ptr_Trans[4] = ptr_T1[5];
    ptr_Trans[5] = ptr_T1[7] + T_b * t;
    ptr_Trans[6] = 0;
    ptr_Trans[7] = 0;
    ptr_Trans[8] = 1.0f;
    return Trans;
}

Mat Translation_Affine_Matrix(float t, float T_a, float T_b)
{
    Mat Trans(3, 3, CV_32FC1);
    float* ptr_Trans = (float*)Trans.data;

    ptr_Trans[0] = 1.0f;
    ptr_Trans[1] = 0;
    ptr_Trans[2] = T_a * t;
    ptr_Trans[3] = 0;
    ptr_Trans[4] = 1.0f;
    ptr_Trans[5] = T_b * t;
    ptr_Trans[6] = 0;
    ptr_Trans[7] = 0;
    ptr_Trans[8] = 1.0f;

    return Trans;
}
