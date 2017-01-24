#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/nonfree/nonfree.hpp>

#include <cstring>
#include <algorithm>
#include <queue>
#include <string>
#include <iostream>
#include <cstdio>
#include <map>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#define For(i,a,b) for(int i=a;i<=b;i++)
using namespace std;
using namespace cv;
struct fr{
    int id;
    double v;
};
fr A[100000];
char Path1[200];
const double THRESHOLD = 400;
bool Vis[100000],Outline[100000];
int num,T,numInt;
int thresh = 100;
char TimeSpot[50];
int max_thresh = 255;
RNG rng(12345);
Mat Ele[2][17];
double C[20];
int tot;
bool cmp(const fr &a1,const fr &a2)
{
    return a1.v<a2.v;
}
void Split(Mat im, int x)
{
    int Width = im.cols;
    int Height = im.rows;
    int Wst = Width/4;
    int Hst = Height/4;
    int w[5] = {0,Wst,2*Wst,3*Wst,Width};
    int h[5] = {0,Hst,2*Hst,3*Hst,Height};
    int tot = 0;
    For(i,0,3)
    For(j,0,3)
    {
        int zsx = w[i];
        int yxx = w[i+1];
        int zsy = h[j];
        int yxy = h[j+1];
        tot++;
        Mat tmp(im, Range(zsy, yxy),Range(zsx,yxx));
        tmp.copyTo(Ele[x][tot]);
    }
}
    
double CompareHSV(Mat src_test1,Mat src_test2)
{
    Mat hsv_test1;
    Mat hsv_test2;
    /// 转换到 HSV
    cvtColor( src_test1, hsv_test1, CV_BGR2HSV );
    cvtColor( src_test2, hsv_test2, CV_BGR2HSV );
    /// 对hue通道使用30个bin,对saturatoin通道使用32个bin
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };
    // hue的取值范围从0到256, saturation取值范围从0到180
    float h_ranges[] = { 0, 256 };
    float s_ranges[] = { 0, 180 };
    const float* ranges[] = { h_ranges, s_ranges };
    
    // 使用第0和第1通道
    int channels[] = { 0, 1 };
    
    /// 直方图
    MatND hist_test1;
    MatND hist_test2;
    
    /// 计算HSV图像的直方图
    calcHist( &hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false );
    normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );
    
    calcHist( &hsv_test2, 1, channels, Mat(), hist_test2, 2, histSize, ranges, true, false );
    normalize( hist_test2, hist_test2, 0, 1, NORM_MINMAX, -1, Mat() );
    
    double Similarity = compareHist( hist_test2, hist_test1, 0);
    return Similarity;
}
/**
 * Calculate euclid distance
 */
double euclidDistance(Mat& vec1, Mat& vec2) {
    double sum = 0.0;
    int dim = vec1.cols;
    for (int i = 0; i < dim; i++) {
        sum += (vec1.at<uchar>(0,i) - vec2.at<uchar>(0,i)) * (vec1.at<uchar>(0,i) - vec2.at<uchar>(0,i));
    }
    return sqrt(sum);
}

/**
 * Find the index of nearest neighbor point from keypoints.
 */
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
    int neighbor = -1;
    double minDist = 1e6;
    
    for (int i = 0; i < descriptors.rows; i++) {
        KeyPoint pt = keypoints[i];
        Mat v = descriptors.row(i);
        double d = euclidDistance(vec, v);
        //printf("%d %f\n", v.cols, d);
        if (d < minDist) {
            minDist = d;
            neighbor = i;
        }
    }
    
    if (minDist < THRESHOLD) {
        return neighbor;
    }
    
    return -1;
}

/**
 * Find pairs of points with the smallest distace between them
 */
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
               vector<KeyPoint>& keypoints2, Mat& descriptors2,
               vector<Point2f>& srcPoints, vector<Point2f>& dstPoints) {
    for (int i = 0; i < descriptors1.rows; i++) {
        KeyPoint pt1 = keypoints1[i];
        Mat desc1 = descriptors1.row(i);
        int nn = nearestNeighbor(desc1, keypoints2, descriptors2);
        if (nn >= 0) {
            KeyPoint pt2 = keypoints2[nn];
            srcPoints.push_back(pt1.pt);
            dstPoints.push_back(pt2.pt);
        }
    }
}

double CompareSift(Mat img1,Mat frame)
{
    Mat originalGrayImage;
    cvtColor(img1, originalGrayImage, CV_BGR2GRAY);
    // equalizeHist(originalGrayImage,originalGrayImage);
    // initialize detector and extractor
    FeatureDetector* detector;
    detector = new SiftFeatureDetector(
                                       0, // nFeatures
                                       4, // nOctaveLayers
                                       0.04, // contrastThreshold
                                       10, //edgeThreshold
                                       1.6 //sigma
                                       );
    
    DescriptorExtractor* extractor;
    extractor = new SiftDescriptorExtractor();
    
    // Compute keypoints and descriptor from the source image in advance
    vector<KeyPoint> keypoints2;
    Mat descriptors2;
    detector->detect(originalGrayImage, keypoints2);
    extractor->compute(originalGrayImage, keypoints2, descriptors2);
    //   printf("original image:%d keypoints are found.\n", (int)keypoints2.size());
    Size size = frame.size();
    Mat grayFrame(size, CV_8UC1);
    cvtColor(frame, grayFrame, CV_BGR2GRAY);
    //equalizeHist(grayFrame,grayFrame);
    vector<KeyPoint> keypoints1;
    Mat descriptors1;
    vector<DMatch> matches;
    // Detect keypoints
    detector->detect(grayFrame, keypoints1);
    extractor->compute(grayFrame, keypoints1, descriptors1);
    for (int i=0; i<keypoints1.size(); i++)
    {
        KeyPoint kp = keypoints1[i];
        
    }
    // Find nearest neighbor pairs
    vector<Point2f> srcPoints;
    vector<Point2f> dstPoints;
    findPairs(keypoints1, descriptors1, keypoints2, descriptors2, srcPoints, dstPoints);
    return srcPoints.size()/(double)keypoints1.size();
    
}

double CompareaHash(Mat img,Mat img2)
{
    Mat gray1,gray2;
    int Fip1[12][12], Fip2[12][12];
    double Ave1,Ave2;
    memset(Fip1,0,sizeof(Fip1));
    memset(Fip2,0,sizeof(Fip2));
    resize(img,gray1,Size(8,8),0,0,CV_INTER_LINEAR);
    cvtColor(gray1,gray1,CV_BGR2GRAY);
    resize(img2,gray2,Size(8,8),0,0,CV_INTER_LINEAR);
    cvtColor(gray2,gray2,CV_BGR2GRAY);
    int Hamming = 0;
    For(i,0,7)
    {
        uchar* P1 = gray1.ptr<uchar>(i);
        uchar* P2 = gray2.ptr<uchar>(i);
        For(j,0,7)
        {
            Ave1+=P1[j];
            Ave2+=P2[j];
        }
    }
    Ave1 = Ave1/(double)64;
    Ave2 = Ave2/(double)64;
    For(i,0,7)
    {
        uchar* P1 = gray1.ptr<uchar>(i);
        uchar* P2 = gray2.ptr<uchar>(i);
        For(j,0,7)
        {
            if(P1[j]>Ave1) Fip1[i][j] = 1;
            if(P2[j]>Ave1) Fip2[i][j] = 1;
            Hamming+=(Fip1[i][j]^Fip2[i][j]);
        }
    }
    return ((64-Hamming)/(double)64);
}

double ComparedHash(Mat img,Mat img2)
{
    Mat gray1,gray2;
    int Fip1[12][12], Fip2[12][12];
    memset(Fip1,0,sizeof(Fip1));
    memset(Fip2,0,sizeof(Fip2));
    resize(img,gray1,Size(9,8),0,0,CV_INTER_LINEAR);
    cvtColor(gray1,gray1,CV_BGR2GRAY);
    resize(img2,gray2,Size(9,8),0,0,CV_INTER_LINEAR);
    cvtColor(gray2,gray2,CV_BGR2GRAY);
    int Hamming = 0;
    For(i,0,7)
    {
        uchar* P1 = gray1.ptr<uchar>(i);
        uchar* P2 = gray2.ptr<uchar>(i);
        For(j,0,7)
        {
            int tmp1 = P1[j] - P1[j+1];
            int tmp2 = P2[j] - P2[j+1];
            if(tmp1>0) Fip1[i][j] = 1;
            if(tmp2>0) Fip2[i][j] = 1;
            Hamming += (Fip1[i][j]^Fip2[i][j]);
        }
    }
    return ((64-Hamming)/(double)64);
}
double CompareHSV_Pieces(Mat im1,Mat im2)
{
    double sum = 0;
    memset(C,0,sizeof(C));
    Split(im1,0);
    Split(im2,1);
    For(i,1,16)
    {
        double Sim = CompareHSV(Ele[0][i],Ele[1][i]);
        C[i] = Sim;
    }
    sort(C+1,C+16);
    For(i,3,14)
    sum+=C[i];
    return sum/12;
}

double Compare(Mat im1,Mat im2)
{
    return (ComparedHash(im1,im2));
}

Mat Draw(Mat result, Mat src)
{
    int S = 0;
    int id = 0;
    Mat threshold_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    threshold( result, threshold_output, thresh, 255, THRESH_BINARY );
    findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );// 找到轮廓
    vector<vector<Point> > contours_poly( contours.size() );   // 多边形逼近轮廓
    vector<Rect> boundRect( contours.size() );  // 矩形外接多边形
    for( int i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
    }
    Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )  // 寻找最大矩形框
    {
          int W = boundRect[i].br().x-boundRect[i].tl().x;
        int H = boundRect[i].br().y-boundRect[i].tl().y;
        if(S<W*H && W>H && W>0 && H>0)
        {
            S = W*H;
            id = i;
        }       
    }
    Mat F = Mat::zeros(1,1, CV_8UC1);
    if(S<300000) return F;
    Mat tmp(src, Range(boundRect[id].tl().y,boundRect[id].br().y),Range(boundRect[id].tl().x,boundRect[id].br().x));
    return tmp;
}

Mat GetROI(Mat src)
{
    Mat result;
    Mat src_gray;
    cvtColor(src, src_gray, CV_BGR2GRAY );
   // equalizeHist( src_gray, src_gray );
    blur( src_gray, src_gray, Size(3,3) );
    Canny(src_gray, result, 150, 450);
    return Draw(result, src);
}
void DispTime(int Sec)
{
  int s = Sec%60;
  int m = (Sec/60)%60;
  int h = (Sec/60)/60;
  sprintf(TimeSpot,"%02d-%02d-%02d", h, m, s);
}

int main(int argc, char** argv)
{
    string VideoPath = "ClassDuanT.mp4";
    Mat Roi1,Roi2;
    freopen("DurationT.txt","r",stdin);
    freopen("DivideAdvice.txt","w",stdout);
    VideoCapture cap(VideoPath);
    scanf("%d", &T);
    Mat frame;
    cap >> frame;
    Roi1 = GetROI(frame);
    while(Roi1.rows==1)
    {
        cap >> frame;
        Roi1 = GetROI(frame);
    }
    num = 1;
    imwrite("OutlineK/(1).jpg",frame);
    double ans;
    while (1)
    {
        cap >> frame;
        if(frame.empty()) break;
        tot++;
        if(tot%10!=0) continue;
        Roi2 = GetROI(frame);
        if(Roi2.rows==1) continue;
        ans = Compare(Roi1,Roi2);
        if(ans<=0.75 && (tot-A[num].id)>=30)
        {
            A[++num].id = tot;
            A[num].v = ans;
            sprintf(Path1, "OutlineK/(%d).jpg", num);
            imwrite(Path1,frame);
        }
        Roi2.copyTo(Roi1);
        
    }
    int FrameALL = tot;
    For(i,1,num)
    {
        int Sec = (A[i].id*T/(double)FrameALL);
        DispTime(Sec);
        printf("#Point%d At%s Sim=%lf\n", i, TimeSpot, A[i].v);
    }
    return 0;
}




