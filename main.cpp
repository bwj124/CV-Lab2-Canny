#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat srcImg;
Mat image_gray;
Mat dst, detected_edges;
Mat edge;

int edgeThreshold = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int the_ratio = 3;
int kernel_size = 3;



void Gauss_kernel(int kernel_size, double sigma, Mat& kernel);//���ɸ�˹������
void SobleGradDirection(const Mat imageSource, Mat& imageX, Mat& imageY, Mat& gradXY, Mat& theta);//Soble����xy�����ݶ�
void NonLocalMaxValue(const Mat imageInput, Mat& imageOutput, const Mat& theta, const Mat& imageX, const Mat& imageY);//���ݶȷ�ֵ���зǼ���ֵ����
void DoubleThreshold(Mat& iamgeInput, const double lowThreshold, const double highThreshold);//˫��ֵ�㷨
void DoubleThresholdLink(Mat& imageInput, double lowTh, double highTh);//����
//void Gauss_my();

int main()
{
    
    
    Mat image = imread("E:/bwj/ͼƬ/��ֽ/Win10��ֽ/100.jpg");
    imshow("origin image", image);

    //ת��Ϊ�Ҷ�ͼ
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    imshow("gray image", grayImage);

    //��˹�˲�
    Mat gausKernel;
    int kernel_size = 5;
    double sigma = 1;
    Gauss_kernel(kernel_size, sigma, gausKernel);
    Mat gausImage;
    filter2D(grayImage, gausImage, grayImage.depth(), gausKernel);
    imshow("gaus image", gausImage);

    //����XY�����ݶ�
    Mat imageX, imageY, imageXY;
    Mat theta;
    SobleGradDirection(gausImage, imageX, imageY, imageXY, theta);
    imshow("XY grad", imageXY);

    //���ݶȷ�ֵ���зǼ���ֵ����
    Mat localImage;
    NonLocalMaxValue(imageXY, localImage, theta, imageX, imageY);;
    imshow("Non local maxinum image", localImage);

    //˫��ֵ�㷨���ͱ�Ե����
    DoubleThreshold(localImage, 60, 100);
    DoubleThresholdLink(localImage, 60, 100);
    imshow("canny image", localImage);

    Mat opencvCanny;
    Canny(image, opencvCanny, 80, 200);
    imshow("opencv canny", opencvCanny);
    imwrite("E:/bwj/ͼƬ/��ֽ/Win10��ֽ/100-0.jpg", opencvCanny);

    waitKey(0);

    

    cout << "end" << endl;
    return 0;
}
  
void Gauss_kernel(int kernel_size, double sigma, Mat& kernel)
{
    const double PI = 4.0 * atan(1.0);
    int center = kernel_size / 2;
    double sum = 0;

    kernel = Mat(kernel_size, kernel_size, CV_32FC1);
    float s = 2 * sigma * sigma;
    for (int i = 0; i < kernel_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
        {
            int x = i - center;
            int y = j - center;

            kernel.at<float>(i, j) = (1 / (2 * PI * s)) * exp(-((x * x + y * y) / s));
            sum += kernel.at<float>(i, j);
        }
    }

    for (int i = 0; i < kernel_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
        {
            kernel.at<float>(i, j) /= sum;
            cout << kernel.at<float>(i, j) << "  ";
        }
        cout << endl << endl;
    }
    return;
}



/*
�����ݶ�ֵ�ͷ���
imageSource ԭʼ�Ҷ�ͼ
imageX X�����ݶ�ͼ��
imageY Y�����ݶ�ͼ��
gradXY �õ���ݶȷ�ֵ
pointDirection �ݶȷ���Ƕ�
*/

void SobleGradDirection(const Mat imageSource, Mat& imageX, Mat& imageY, Mat& gradXY, Mat& theta)
{
    imageX = Mat::zeros(imageSource.size(), CV_32SC1);
        imageY = Mat::zeros(imageSource.size(), CV_32SC1);
        gradXY = Mat::zeros(imageSource.size(), CV_32SC1);
        theta = Mat::zeros(imageSource.size(), CV_32SC1);
    
        int rows = imageSource.rows;
        int cols = imageSource.cols;
    
        int stepXY = imageX.step;
        int step = imageSource.step;
        /*
        Mat.step����ָͼ���һ��ʵ��ռ�õ��ڴ泤�ȣ�
        */
        uchar* PX = imageX.data;
        uchar* PY = imageY.data;
        uchar* P = imageSource.data;
        uchar* XY = gradXY.data;

    for (int i = 1; i < rows - 1; i++)
    {
        for (int j = 1; j < cols - 1; j++)
        {
            int a00 = P[(i - 1) * step + j - 1];
            int a01 = P[(i - 1) * step + j];
            int a02 = P[(i - 1) * step + j + 1];

            int a10 = P[i * step + j - 1];
            int a11 = P[i * step + j];
            int a12 = P[i * step + j + 1];

            int a20 = P[(i + 1) * step + j - 1];
            int a21 = P[(i + 1) * step + j];
            int a22 = P[(i + 1) * step + j + 1];

          
            double gradX = double(a02 + 2 * a12 + a22 - a00 - 2 * a10 - a20);
            double gradY = double(a20 + 2 * a21 + a22 - a00 - 2 * a01 - a02);
            imageX.at<int>(i, j) = abs(gradX);
            imageY.at<int>(i, j) = abs(gradY);
            if (gradX == 0)
            {
                gradX = 0.000000000001;
            }

            theta.at<int>(i, j) = atan(gradY / gradX) * 57.3;
            theta.at<int>(i, j) = (theta.at<int>(i, j) + 360) % 360;
            gradXY.at<int>(i, j) = sqrt(gradX * gradX + gradY * gradY);
        }

        
       

    }
    convertScaleAbs(imageX, imageX);//תΪ8bitͼ��
    convertScaleAbs(imageY, imageY);
    convertScaleAbs(gradXY, gradXY);
}


/*
�ֲ��Ǽ���ֵ����
���Ÿõ��ݶȷ��򣬱Ƚ�ǰ��������ķ�ֵ��С�����õ����ǰ�����㣬������
���õ�С��ǰ����������һ�㣬����Ϊ0��
imageInput ����õ��ݶ�ͼ��
imageOutput ����ķǼ���ֵ����ͼ��
theta ÿ�����ص���ݶȷ���Ƕ�
imageX X�����ݶ�
imageY Y�����ݶ�
*/

void NonLocalMaxValue(const Mat imageInput, Mat& imageOutput, const Mat& theta, const Mat& imageX, const Mat& imageY)
{
    imageOutput = imageInput.clone();


    int cols = imageInput.cols;
    int rows = imageInput.rows;

    for (int i = 1; i < rows - 1; i++)
    {
        for (int j = 1; j < cols - 1; j++)
        {
            if (0 == imageInput.at<uchar>(i, j))continue;

            int g00 = imageInput.at<uchar>(i - 1, j - 1);
            int g01 = imageInput.at<uchar>(i - 1, j);
            int g02 = imageInput.at<uchar>(i - 1, j + 1);

            int g10 = imageInput.at<uchar>(i, j - 1);
            int g11 = imageInput.at<uchar>(i, j);
            int g12 = imageInput.at<uchar>(i, j + 1);

            int g20 = imageInput.at<uchar>(i + 1, j - 1);
            int g21 = imageInput.at<uchar>(i + 1, j);
            int g22 = imageInput.at<uchar>(i + 1, j + 1);

            int direction = theta.at<int>(i, j); //�õ��ݶȵĽǶ�ֵ
            int g1 = 0;
            int g2 = 0;
            int g3 = 0;
            int g4 = 0;
            double tmp1 = 0.0; //���������ص��ֵ�õ��ĻҶ���
            double tmp2 = 0.0;
            double weight = fabs((double)imageY.at<uchar>(i, j) / (double)imageX.at<uchar>(i, j));

            if (weight == 0) 
            {
                weight = 0.0000001;
            }
            if (weight > 1)
            {
                weight = 1 / weight;
            }
            if ((0 <= direction && direction < 45) || 180 <= direction && direction < 225)
            {
                tmp1 = g10 * (1 - weight) + g20 * (weight);
                tmp2 = g02 * (weight)+g12 * (1 - weight);
            }
            if ((45 <= direction && direction < 90) || 225 <= direction && direction < 270)
            {
                tmp1 = g01 * (1 - weight) + g02 * (weight);
                tmp2 = g20 * (weight)+g21 * (1 - weight);
            }
            if ((90 <= direction && direction < 135) || 270 <= direction && direction < 315)
            {
                tmp1 = g00 * (weight)+g01 * (1 - weight);
                tmp2 = g21 * (1 - weight) + g22 * (weight);
            }
            if ((135 <= direction && direction < 180) || 315 <= direction && direction < 360)
            {
                tmp1 = g00 * (weight)+g10 * (1 - weight);
                tmp2 = g12 * (1 - weight) + g22 * (weight);
            }

            if (imageInput.at<uchar>(i, j) < tmp1 || imageInput.at<uchar>(i, j) < tmp2)
            {
                imageOutput.at<uchar>(i, j) = 0;
            }
        }
    }

}


//˫��ֵ�Ļ����ǣ�
//ָ��һ������ֵA��һ������ֵB��һ��ȡBΪͼ������Ҷȼ��ֲ���70%����BΪ1.5��2����С��A��
//�Ҷ�ֵС��A�ģ���Ϊ0,�Ҷ�ֵ����B�ģ���Ϊ255��

void DoubleThreshold(Mat& iamgeInput, const double lowThreshold, const double highThreshold)
{
    int cols = iamgeInput.cols;
    int rows = iamgeInput.rows;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++) {
            double temp = iamgeInput.at<uchar>(i, j);
            temp = temp > highThreshold ? (255) : (temp);
            temp = temp < lowThreshold ? (0) : (temp);
            iamgeInput.at<uchar>(i, j) = temp;
        }
    }

}

/*
���Ӵ���:
�Ҷ�ֵ����A��B֮��ģ���������ص��ٽ���8�����Ƿ��лҶ�ֵΪ255�ģ�
��û��255�ģ���ʾ����һ�������ľֲ�����ֵ�㣬�����ų�����Ϊ0��
����255�ģ���ʾ����һ����������Ե�С��������Ŀ���֮�ģ���Ϊ255��
֮���ظ�ִ�иò��裬ֱ��������֮��һ�����ص㡣

����1��imageInput�������������ݶ�ͼ��
����2��lowTh:����ֵ
����3��highTh:����ֵ
*/
void DoubleThresholdLink(Mat& imageInput, double lowTh, double highTh)
{
    int cols = imageInput.cols;
    int rows = imageInput.rows;

    for (int i = 1; i < rows - 1; i++)
    {
        for (int j = 1; j < cols - 1; j++)
        {
            double pix = imageInput.at<uchar>(i, j);
            if (pix != 255)continue;
            bool change = false;
            for (int k = -1; k <= 1; k++)
            {
                for (int u = -1; u <= 1; u++)
                {
                    if (k == 0 && u == 0)continue;
                    double temp = imageInput.at<uchar>(i + k, j + u);
                    if (temp >= lowTh && temp <= highTh)
                    {
                        imageInput.at<uchar>(i + k, j + u) = 255;
                        change = true;
                    }
                }
            }
            if (change)
            {
                if (i > 1)i--;
                if (j > 2)j -= 2;

            }
        }
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (imageInput.at<uchar>(i, j) != 255)
            {
                imageInput.at<uchar>(i, j) = 0;
            }
        }
    }
}