#include "KFDenoising.h"

KFDenoising::KFDenoising(int h, int w, bool s_d, bool t_d)
{
    /************** warm up **************/
    Mat img1 = imread("./img/origin0.jpg", 0);
    Mat img1_gaussian, img1_canny;
    GaussianBlur(img1, img1_gaussian, Size(5, 5), 3, 3);
    Canny(img1_gaussian, img1_canny, 150, 200);
    start_dynamic = s_d;
    stay = !start_dynamic;
    height = h;
    width = w;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            p_img.push_back(1000);
            q_img.push_back(0);
            r_img.push_back(2000);
            k_img.push_back(0);
        }
    }

    count = 0;
    start = true;
    tip_down = t_d;

    mask_predict = Mat::zeros(height, width, CV_8UC1);
    mask_observed = Mat::zeros(height, width, CV_8UC1);

    features_distorted_prev.push_back(Vec2i(0, 0));
    features_distorted_prev.push_back(Vec2i(0, 0));
    features_distorted_prev.push_back(Vec2i(0, 0));
    features_prev.push_back(Vec2i(0, 0));
    features_prev.push_back(Vec2i(0, 0));
    features_prev.push_back(Vec2i(0, 0));

    initialize_mat_motion();
}

Mat KFDenoising::run(Mat img_updated, float T_a, float T_b)
{
    clock_t start_time, end_time;
    start_time = clock();
    float s = sqrt(T_a * T_a + T_b * T_b);
    Mat img_distorted_spatial(img_updated.size(), CV_8UC1, Scalar::all(0));
    Mat img_spatial(img_updated.size(), CV_8UC1, Scalar::all(0));
    Mat img_temporal(img_updated.size(), CV_8UC1, Scalar::all(0));

    vector<float> controller_parameters;

    controller_parameters.push_back(T_a);
    controller_parameters.push_back(T_b);

    if (start) {
        if (start_dynamic) {
            start = false;
            img_distorted_spatial = spatial_denoising(img_updated);
            now_distorted_tip = find_tip(img_distorted_spatial);
            cout << "tip:" << now_distorted_tip << endl;
            now_tip = remove_distortion_tip(now_distorted_tip, T_a, T_b);
            cout << "now tip:" << now_distorted_tip << endl;
            features_distorted_now = find_feature_point(img_distorted_spatial);

            if (theta == 0) {
                img_spatial = remove_distortion_img(img_distorted_spatial, T_a, T_b);
            }
            else {
                img_spatial = remove_distortion_img(img_distorted_spatial, controller_parameters);
            }
            img_prev = img_updated.clone();
            //img_prev_denoised = img_spatial.clone();
            img_spatial.convertTo(img_spatial, CV_32FC1);
            img_prev_denoised = img_spatial.clone();
            img_spatial.convertTo(img_spatial, CV_8UC1);
            update_distortion_information();
            mask_predict = mask_observed.clone();
            return img_spatial;
        }
        else {
            cout << "start" << endl;
            start = false;
            img_distorted_spatial = spatial_denoising(img_updated);
            count++;
            now_tip = find_tip(img_distorted_spatial);
            now_distorted_tip = find_tip(img_distorted_spatial);
            features_distorted_now = find_feature_point(img_distorted_spatial);
            features_now = features_distorted_now;
            img_prev = img_updated.clone();
            //img_prev_denoised = img_distorted_spatial.clone();
            img_distorted_spatial.convertTo(img_distorted_spatial, CV_32FC1);
            img_prev_denoised = img_distorted_spatial.clone();
            img_distorted_spatial.convertTo(img_distorted_spatial, CV_8UC1);
            mask_predict = Mat::ones(height, width, CV_8UC1);
            update_distortion_information();
            return img_distorted_spatial;
        }
    }

    if (theta == 0 && s < 1.0f) {
        if (stay) {
            cout << "stay!" << endl;
            stay = true;
            count++;
            if (count > 60) {
                img_distorted_spatial = img_updated.clone();
            }
            else {
                img_distorted_spatial = spatial_denoising(img_updated);
            }
            mask_observed = Mat::ones(height, width, CV_8UC1);
            //Mat Trans = Trans_Matrix(a, x0, b, y0, c, z0, theta, 1.0f, T_a, T_b);
            //Mat Trans_f = Trans_feature(features_prev, features_now);
            now_tip = find_tip(img_distorted_spatial);
            now_distorted_tip = find_tip(img_distorted_spatial);
            //features_distorted_now = find_feature_point(img_distorted_spatial);
            features_distorted_now = features_distorted_now;
            remove_distortion_features(features_now, controller_parameters);
            Mat T = Translation_Affine_Matrix(1, 0, 0);
            Mat img_translated = Translate_img(img_prev_denoised, T);
            Kalman_Img_update(img_prev_denoised, img_distorted_spatial);
            img_temporal = Kalman_Img(img_translated, img_distorted_spatial);
            img_prev = img_updated.clone();
            img_prev_denoised = img_temporal.clone();
            img_temporal.convertTo(img_temporal, CV_8UC1);
            update_distortion_information();
            return img_temporal;
        }
    }
    cout << "dynamic" << endl;
    stay = false;
    count++;
    if (count > 60) {
        img_distorted_spatial = img_updated.clone();
    }
    else {
        img_distorted_spatial = spatial_denoising(img_updated);
    }
    mask_observed = Mat::ones(height, width, CV_8UC1);
    now_distorted_tip = find_tip(img_distorted_spatial);
    features_distorted_now = find_feature_point(img_distorted_spatial);
    //drawCircle(img_updated, features_distorted_now);
    remove_distortion_features(features_now, controller_parameters);
    Vec2f Translation_f;
    Translation_f = Translation_tip(prev_distorted_tip, now_distorted_tip, height);

    //cout << now_distorted_tip << endl;
    now_tip = remove_distortion_tip(now_distorted_tip, Translation_f[0], Translation_f[1]);
    //cout << now_tip << endl;
    //cout << prev_tip << endl;
    if (theta == 0) {
        img_spatial = remove_distortion_img(img_distorted_spatial, Translation_f[0], Translation_f[1]);
    }
    else {
        img_spatial = remove_distortion_img(img_distorted_spatial, controller_parameters);
    }
    //imshow("spatial", img_spatial);
    //waitKey(0);
    Mat Trans;
    Mat Trans_f;
    if (theta == 0) {
        Trans = Translation_Affine_Matrix(1, T_a, T_b).clone();
        Trans_f = Translation_Affine_Matrix_tip(prev_tip, now_tip, height).clone();
    }
    else {
        Trans = Trans_Matrix(1, 1, 1, 1, 1, 1, theta, 1.0f, T_a, T_b);
        Trans_f = Trans_feature(features_prev, features_now).clone();
    }
    Mat T(3, 3, CV_32FC1);
    cout << "Trans:" << Trans << endl;
    cout << "Trans_f:" << Trans_f << endl;
    if (theta == 0) {
        Kalman_Motion_Translation(Trans, Trans_f, T, T_a, T_b);
    }
    else {
        Kalman_Motion(Trans, Trans_f, T, controller_parameters);
    }

    cout << T << endl;
    Mat img_translated = Translate_img(img_prev_denoised, T);
    //imshow("spatial", img_spatial);
    Mat img_translated_uchar;
    img_translated.convertTo(img_translated_uchar, CV_8UC1);
    //imshow("translated", img_translated_uchar);
    imwrite("./spatial" + to_string(count) + ".bmp", img_spatial);
    imwrite("./translated" + to_string(count) + ".bmp", img_translated_uchar);
    //waitKey(0);
    destroyAllWindows();
    Kalman_Img_update(img_translated, img_spatial);

    img_temporal = Kalman_Img(img_translated, img_spatial);
    img_prev = img_updated.clone();
    img_prev_denoised = img_temporal.clone();
    img_temporal.convertTo(img_temporal, CV_8UC1);
    update_distortion_information();
    return img_temporal;
}

Mat KFDenoising::spatial_denoising(Mat img)
{
    clock_t start, end;
    start = clock();
    int height = img.rows;
    int width = img.cols;
    img.convertTo(img, CV_32FC1);
    vector<int> row_idx;
    vector<int> col_idx;
    vector<Mat> blocks;
    vector<Mat> blocks_origin;
    vector<Mat> img_s;
    img_s.push_back(img);
    for (int i = 0; i <= height - 16; i += 16) {
        row_idx.push_back(i);
    }
    for (int j = 0; j <= width - 16; j += 16) {
        col_idx.push_back(j);
    }

    int b_r = row_idx.size();
    int b_c = col_idx.size();

    // threshold for some noises to make the similarity more accurate
    start = clock();

    dct(img_s[0], img_s[0]);
    DetectZero(img_s, 65);
    idct(img_s[0], img_s[0]);

    Mat tmp(16, 16, CV_32FC1);
    start = clock();
    for (int i = 0; i < row_idx.size(); i++) {
        for (int j = 0; j < col_idx.size(); j++) {
            Mat tmp_dct;
            vector<Mat> tmp_dct_list;
            tmp = img_s[0](Rect(col_idx[j], row_idx[i], 16, 16));
            //blocks_origin.push_back(tmp);
            dct(tmp, tmp_dct);
            //tmp_dct_list.push_back(tmp_dct);
            //DetectZero(tmp_dct_list, 60);
            //idct(tmp_dct, tmp);
            blocks_origin.push_back(tmp);
            blocks.push_back(tmp_dct);
        }
    }
    end = clock();
    clock_t duration = 0;
    clock_t duration2 = 0;
    cout << "DCT processing time cost:" << (double)(end - start) / CLOCKS_PER_SEC << endl;
    vector<Mat> data;
    Mat denominator_hd(img.size(), CV_32FC1, Scalar::all(0));
    Mat numerator_hd(img.size(), CV_32FC1, Scalar::all(0));
    clock_t start1 = clock();
    for (int i = 0; i < row_idx.size(); i++) {
        for (int j = 0; j < col_idx.size(); j++) {
            data.clear();

            // search_window
            int row_min = max(0, i - (3 - 1) / 2);
            int row_max = min(b_r - 1, i + (3 - 1) / 2);
            int row_length = row_max - row_min + 1;

            int col_min = max(0, j - (4 - 1) / 2);
            int col_max = min(b_c - 1, j + (4 - 1) / 2);
            int col_length = col_max - col_min + 1;

            float* distance = new float[row_length * col_length];
            float* weight = new float[row_length * col_length];

            const Mat relevence = blocks_origin[i * b_c + j];
            const Mat relevence_dct = blocks[i * b_c + j];
            float d;
            float sum = 0;
            float alpha = 0.7;
            // calculate the similarity and the weight
            start = clock();
            for (int p = 0; p < row_length; p++)
            {
                for (int q = 0; q < col_length; q++)
                {
                    Mat tmp1, tmp1_dct;
                    tmp1 = blocks_origin[(p + row_min) * b_c + (q + col_min)];
                    tmp1_dct = blocks[(p + row_min) * b_c + (q + col_min)];
                    d = (1 - alpha) * cal_sim(relevence, tmp1);
                    d += alpha * cal_sim(relevence_dct, tmp1_dct);
                    weight[p * col_length + q] = exp(-d / 2000);

                    sum += weight[p * col_length + q];
                }
            }
            end = clock();
            duration += end - start;
            Rect rect;
            rect.x = col_idx[j];
            rect.y = row_idx[i];
            rect.height = 16;
            rect.width = 16;
            Mat window(tmp.size(), CV_32FC1, Scalar::all(1));
            start = clock();
            // weighted stacking
            for (int p = 0; p < row_length; p++)
            {
                for (int q = 0; q < col_length; q++)
                {
                    Mat tmp1;
                    tmp1 = blocks_origin[(p + row_min) * b_c + (q + col_min)];
                    numerator_hd(rect) += weight[p * col_length + q] * tmp1;
                }
            }

            denominator_hd(rect) += sum * window;
            end = clock();
            duration2 += end - start;
            delete[] distance;
            delete[] weight;
        }
    }
    end = clock();
    // normalization
    Mat img_denoised = numerator_hd / denominator_hd;
    img_denoised.convertTo(img_denoised, CV_8UC1);
    return img_denoised;
}

void KFDenoising::DetectZero(vector<Mat>& input, float threshold)
{
    for (int k = 0; k < input.size(); k++) {
        for (int i = 0; i < input[k].rows; i++) {
            for (int j = 0; j < input[k].cols; j++) {
                if (fabs(input[k].at<float>(i, j)) < threshold) {
                    input[k].at<float>(i, j) = 0;
                }
            }
        }
    }
}

float KFDenoising::cal_sim(Mat b1, Mat b2)
{
    int h = b1.rows;
    int w = b1.cols;
    Mat b1_c, b2_c;
    b1_c = b1.clone();
    b2_c = b2.clone();
    float* M1 = (float*)b1_c.data;
    float* M2 = (float*)b2_c.data;
    float sim = 0;

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            sim += (M1[i * w + j] - M2[i * w + j]) * (M1[i * w + j] - M2[i * w + j]);
        }
    }
    return sim;
}

void KFDenoising::remove_distortion_features(vector<Vec2i>& features, vector<float> controller_parameters)
{
    vector<Mat> Trans_Matrix_list;
    for (int i = 0; i < features.size(); i++) {
        float t = features[i][0] / height;
        Trans_Matrix_list.push_back(Trans_Matrix(p[0], p[1], p[2], p[3], p[4], p[5], p[6], 1 - t, p[7], p[8]));
    }
    float* ptr_trans_blur;
    float m_blur = 0.0f;
    float n_blur = 0.0f;
    float p_blur = 0.0f;
    float q_blur = 0.0f;
    float a_blur = 0.0f;
    float b_blur = 0.0f;
    int x_blur = 0;
    int y_blur = 0;
    for (int i = 0; i < features.size(); i++) {
        ptr_trans_blur = (float*)Trans_Matrix_list[i].data;
        m_blur = ptr_trans_blur[0];
        n_blur = ptr_trans_blur[3];
        p_blur = ptr_trans_blur[1];
        q_blur = ptr_trans_blur[4];
        a_blur = ptr_trans_blur[2];
        b_blur = ptr_trans_blur[5];
        x_blur = m_blur * features[i][0] + p_blur * features[i][1] + a_blur;
        y_blur = n_blur * features[i][0] + q_blur * features[i][1] + b_blur;
        features[i][0] = x_blur;
        features[i][1] = y_blur;
    }
}

Vec2i KFDenoising::remove_distortion_tip(Vec2i tip, float x, float y)
{
    Mat Trans_Matrix;
    float t = tip[0];
    t /= height;
    Trans_Matrix = Translation_Affine_Matrix(1 - t, x, y);
    float* ptr_trans_blur;
    float m_blur, n_blur, p_blur, q_blur, a_blur, b_blur, x_blur, y_blur;
    ptr_trans_blur = (float*)Trans_Matrix.data;
    m_blur = ptr_trans_blur[0];
    n_blur = ptr_trans_blur[3];
    p_blur = ptr_trans_blur[1];
    q_blur = ptr_trans_blur[4];
    a_blur = ptr_trans_blur[2];
    b_blur = ptr_trans_blur[5];
    x_blur = m_blur * tip[0] + p_blur * tip[1] + a_blur;
    y_blur = n_blur * tip[0] + q_blur * tip[1] + b_blur;
    tip[0] = x_blur;
    tip[1] = y_blur;
    return tip;
}

Mat KFDenoising::remove_distortion_img(Mat img_distorted, float x, float y)
{
    Mat img_denoised(img_distorted.size(), CV_8UC1, Scalar::all(0));
    uchar* ptr_img_denoised = img_denoised.data;
    uchar* ptr_img_distorted = img_distorted.data;
    vector<Mat> Trans_Matrix_list;
    for (int i = 0; i < height; i++) {
        float t = i;
        t /= height;
        Trans_Matrix_list.push_back(Translation_Affine_Matrix(1 - t, -x, -y));
    }

    uchar* ptr_mask_observed = mask_observed.data;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            ptr_mask_observed[i * width + j] = 0;
        }
    }

    float* ptr_trans_blur;
    float m_blur = 0.0f;
    float n_blur = 0.0f;
    float p_blur = 0.0f;
    float q_blur = 0.0f;
    float a_blur = 0.0f;
    float b_blur = 0.0f;
    int x_blur = 0;
    int y_blur = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            ptr_trans_blur = (float*)Trans_Matrix_list[i].data;
            m_blur = ptr_trans_blur[0];
            n_blur = ptr_trans_blur[3];
            p_blur = ptr_trans_blur[1];
            q_blur = ptr_trans_blur[4];
            a_blur = ptr_trans_blur[2];
            b_blur = ptr_trans_blur[5];
            x_blur = m_blur * i + p_blur * j + a_blur;
            y_blur = n_blur * i + q_blur * j + b_blur;
            if (x_blur >= 0 && x_blur < height && y_blur >= 0 && y_blur < width) {
                ptr_img_denoised[i * width + j] = ptr_img_distorted[x_blur * width + y_blur];
                ptr_mask_observed[i * width + j] = 255;
            }
        }
    }
    return img_denoised;
}

Mat KFDenoising::remove_distortion_img(Mat img_distorted, vector<float> controller_parameters)
{
    Mat img_denoised(img_distorted.size(), CV_8UC1, Scalar::all(0));
    uchar* ptr_img_denoised = img_denoised.data;
    uchar* ptr_img_distorted = img_distorted.data;
    vector<Mat> Trans_Matrix_list;
    for (int i = 0; i < height; i++) {
        float t = i;
        t /= height;
        Trans_Matrix_list.push_back(Trans_Matrix(p[0], p[1], p[2], p[3], p[4], p[5], -p[6], 1 - t, -p[7], -p[8]));
    }
    float* ptr_trans_blur;
    float m_blur = 0.0f;
    float n_blur = 0.0f;
    float p_blur = 0.0f;
    float q_blur = 0.0f;
    float a_blur = 0.0f;
    float b_blur = 0.0f;
    int x_blur = 0;
    int y_blur = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i - p[7]) * 480 / (480 - p[7]);
            //cout << idx << endl;
            ptr_trans_blur = (float*)Trans_Matrix_list[i].data;
            m_blur = ptr_trans_blur[0];
            n_blur = ptr_trans_blur[3];
            p_blur = ptr_trans_blur[1];
            q_blur = ptr_trans_blur[4];
            a_blur = ptr_trans_blur[2];
            b_blur = ptr_trans_blur[5];
            x_blur = m_blur * i + p_blur * j + a_blur;
            y_blur = n_blur * i + q_blur * j + b_blur;
            if (x_blur >= 0 && x_blur < height && y_blur >= 0 && y_blur < width) {
                ptr_img_denoised[i * width + j] = ptr_img_distorted[x_blur * width + y_blur];
            }
        }
    }
    return img_denoised;
}

Mat KFDenoising::Translate_img(Mat img_prev, Mat T)
{
    Mat Trans_inverse;
    Mat img_denoised(height, width, CV_32FC1);
    invert(T, Trans_inverse);
    float* ptr_Trans_inverse = (float*)Trans_inverse.data;
    float* ptr_Trans = (float*)T.data;
    int x_distorted = 0;
    int y_distorted = 0;

    Mat mask_predict_new = Mat::zeros(height, width, CV_8UC1);
    uchar* ptr_mask_predict_new = mask_predict_new.data;
    uchar* ptr_mask_predict = mask_predict.data;

    float* ptr_prev_img_denoised = (float*)img_prev.data;
    float* ptr_img_denoised = (float*)img_denoised.data;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            x_distorted = i * ptr_Trans_inverse[0] + j * ptr_Trans_inverse[1] + ptr_Trans_inverse[2];
            y_distorted = i * ptr_Trans_inverse[3] + j * ptr_Trans_inverse[4] + ptr_Trans_inverse[5];
            if (x_distorted >= 0 && x_distorted < height && y_distorted >= 0 && y_distorted < width) {
                ptr_img_denoised[i * width + j] = ptr_prev_img_denoised[x_distorted * width + y_distorted];
                if (ptr_mask_predict[x_distorted * width + y_distorted] != 0) {
                    ptr_mask_predict_new[i * width + j] = 255;
                }
            }
            else {
                ptr_img_denoised[i * width + j] = 0;
            }
        }
    }
    mask_predict = mask_predict_new.clone();
    return img_denoised;
}

void KFDenoising::update_distortion_information()
{
    prev_tip[0] = now_tip[0];
    prev_tip[1] = now_tip[1];
    prev_distorted_tip[0] = now_distorted_tip[0];
    prev_distorted_tip[1] = now_distorted_tip[1];
    for (int i = 0; i < features_distorted_now.size(); i++) {
        features_distorted_prev[i][0] = features_distorted_now[i][0];
        features_distorted_prev[i][1] = features_distorted_now[i][1];
    }
    for (int i = 0; i < features_now.size(); i++) {
        features_prev[i][0] = features_now[i][0];
        features_prev[i][1] = features_now[i][1];
    }
}

void KFDenoising::Kalman_Motion_Translation(Mat Trans, Mat Trans_f, Mat& T, float a1, float a2)
{
    Mat Translation_predict(2, 1, CV_32FC1);
    Mat Translation_observed(2, 1, CV_32FC1);
    Mat Translation(2, 1, CV_32FC1);
    float* ptr_predict, * ptr_observed, * ptr_translation;
    float* ptr_trans, * ptr_trans_f, * ptr_T;
    ptr_predict = (float*)Translation_predict.data;
    ptr_observed = (float*)Translation_observed.data;
    ptr_translation = (float*)Translation.data;
    ptr_trans = (float*)Trans.data;
    ptr_trans_f = (float*)Trans_f.data;
    ptr_T = (float*)T.data;
    ptr_predict[0] = ptr_trans[2];
    ptr_predict[1] = ptr_trans[5];
    ptr_observed[0] = ptr_trans_f[2];
    ptr_observed[1] = ptr_trans_f[5];
    Mat inverse(2, 2, CV_32FC1);
    Calculate_Translation_Q(Q_Translation_Motion, a1, a2);
    Calculate_Translation_R(R_Translation_Motion, ptr_trans[2], ptr_trans[5]);

    invert((Q_Translation_Motion + R_Translation_Motion), inverse);
    K_Translation_Motion = Q_Translation_Motion * inverse;
    cout << "K:" << K_Translation_Motion << endl;
    Translation = Translation_predict + K_Translation_Motion * (Translation_observed - Translation_predict);
    ptr_translation = (float*)Translation.data;
    ptr_T[0] = 1.0f;
    ptr_T[1] = 0.0f;
    ptr_T[2] = ptr_translation[0];
    ptr_T[3] = 0.0f;
    ptr_T[4] = 1.0f;
    ptr_T[5] = ptr_translation[1];
    ptr_T[6] = 0.0f;
    ptr_T[7] = 0.0f;
    ptr_T[8] = 1.0f;
    //T = Trans;
}


void KFDenoising::Kalman_Img_update(Mat img_predict, Mat img_observed)
{
    Mat img_gaussian;
    img_observed.convertTo(img_observed, CV_8UC1);
    GaussianBlur(img_observed, img_gaussian, Size(5, 5), 3, 3);
    img_predict.convertTo(img_predict, CV_32FC1);
    img_observed.convertTo(img_observed, CV_32FC1);
    vector<float> sim_list;
    vector<Mat> predict_list;
    vector<Mat> observed_list;
    int block_len = 8;
    for (int i = 0; i < height / block_len; i++) {
        for (int j = 0; j < width / block_len; j++) {
            predict_list.push_back(img_predict(Rect(j * block_len, i * block_len, block_len, block_len)).clone());
            observed_list.push_back(img_observed(Rect(j * block_len, i * block_len, block_len, block_len)).clone());
        }
    }
    for (int i = 0; i < predict_list.size(); i++) {
        sim_list.push_back(cal_sim(predict_list[i], observed_list[i]));
    }
    uchar* ptr_img_gaussian = img_gaussian.data;
    int idx;
    float alpha = 16000.0f;
    float q_sum = 0;
    float k_sum = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            idx = int(i / block_len) * width / block_len + j / block_len;
            if (stay) {
                q_img[i * width + j] = 10;
            }
            else {
                //q_img[i * width + j] = alpha * (1 - exp(-sim_list[idx] / 1280000));
                q_img[i * width + j] = sim_list[idx] / (block_len * block_len);
            }
            //q_img[i * width + j] = alpha * (1 - exp(-sim_list[idx] / 2000));
            q_sum += q_img[i * width + j];
        }
    }
    q_sum /= height * width;
    //cout << "q:" << q_sum << endl;
    float beta = 0.5;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (var_list.size() != 256) {
                r_img[i * width + j] = ptr_img_gaussian[i * width + j] * ptr_img_gaussian[i * width + j] * 0.45 * 0.45 * beta;
                //r_img[i * width + j] = 100;
            }
            else {
                r_img[i * width + j] = var_list[ptr_img_gaussian[i * width + j]];
            }
        }
    }
    for (int i = 0; i < height * width; i++) {
        k_img[i] = (p_img[i] + q_img[i]) / (p_img[i] + q_img[i] + r_img[i] + 0.1);
        p_img[i] = (1 - k_img[i]) * (p_img[i] + q_img[i]);
        k_sum += k_img[i];
    }
    k_sum /= height * width;
}

Mat KFDenoising::Kalman_Img(Mat img_predict, Mat img_observed)
{
    Mat img_denoised(height, width, CV_32FC1);
    //img_denoised.convertTo(img_denoised, CV_32FC1);
    Mat mask_predict_new = Mat::ones(height, width, CV_8UC1);
    uchar* ptr_mask_predict_new = mask_predict_new.data;

    float* ptr_img_denoised = (float*)img_denoised.data;
    float* ptr_predict = (float*)img_predict.data;
    uchar* ptr_observed = img_observed.data;
    uchar* ptr_mask_predict = mask_predict.data;
    uchar* ptr_mask_observed = mask_observed.data;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (ptr_mask_predict[i * width + j] != 0 && ptr_mask_observed[i * width + j] != 0) {
                ptr_img_denoised[i * width + j] = ptr_predict[i * width + j] + k_img[i * width + j] * (ptr_observed[i * width + j] - ptr_predict[i * width + j]);
                ptr_mask_predict_new[i * width + j] = 255;
            }
            else if (ptr_mask_predict[i * width + j] != 0) {
                ptr_img_denoised[i * width + j] = ptr_predict[i * width + j];
                ptr_mask_predict_new[i * width + j] = 255;
            }
            else if (ptr_mask_observed[i * width + j] != 0) {
                ptr_img_denoised[i * width + j] = ptr_observed[i * width + j];
                ptr_mask_predict_new[i * width + j] = 255;
            }
            else {
                ptr_img_denoised[i * width + j] = 0;
                ptr_mask_predict_new[i * width + j] = 0;
            }
        }
    }
    //img_denoised.convertTo(img_denoised, CV_8UC1);
    mask_predict = mask_predict_new.clone();
    return img_denoised;
}

void KFDenoising::initialize_mat_motion()
{
    P_motion = Mat::zeros(6, 6, CV_32FC1);
    Q_motion = Mat::zeros(6, 6, CV_32FC1);
    R_motion = Mat::zeros(6, 6, CV_32FC1);
    K_motion = Mat::zeros(6, 6, CV_32FC1);
    I_motion = Mat::zeros(6, 6, CV_32FC1);
    Q_Translation_Motion = Mat::zeros(2, 2, CV_32FC1);
    R_Translation_Motion = Mat::zeros(2, 2, CV_32FC1);
    K_Translation_Motion = Mat::zeros(2, 2, CV_32FC1);
    float* ptr_i = (float*)I_motion.data;
    int count = 0;
    for (int i = 0; i < 6; i++) {
        ptr_i[count] = 1.0f;
        count += 7;
    }
}

Vec2i KFDenoising::find_tip(Mat img)
{
    return find_tip_feature(img, tip_down);
}
