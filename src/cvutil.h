#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "point.h"
#include "LowPoly.h"
#include "triangle.h"

using namespace std;

cv::Mat getGrad(cv::Mat img)
{
    // convert img to gray scale, do sobel filtering, and normalize
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, CV_BGR2GRAY);
    int scale = 1;
    int delta = 0;

    cv::Mat gradX, gradY, grad;
    cv::Sobel(imgGray, gradX, CV_32F, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
    gradX = cv::abs(gradX);
    cv::Sobel(imgGray, gradY, CV_32F, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
    gradY = cv::abs(gradY);
    cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, grad);
    double minGrad, maxGrad;
    cv::minMaxLoc(grad, &minGrad, &maxGrad);
    grad = grad / maxGrad;

    return grad;
}

vector<Point> selectVertices(cv::Mat &grad, float edgeP, float nonEdgeP, float boundP, float edgeThresh, int &numVertices)
{
    int rows = grad.rows;
    int cols = grad.cols;
    vector<Point> vertices;
    int trueNumVertices = 0;

    // four corners
    Point p1, p2, p3, p4;
    p1.x = 0;
    p1.y = 0;
    p2.x = 0;
    p2.y = rows - 1;
    p3.x = cols - 1;
    p3.y = 0;
    p4.x = cols - 1;
    p4.y = rows - 1;
    vertices.push_back(p1);
    vertices.push_back(p2);
    vertices.push_back(p3);
    vertices.push_back(p4);
    trueNumVertices += 4;

    // boundaries
    for (int row = 1; row < rows-1; row++)
    {
        double randNum = ((double) rand() / (RAND_MAX));
        if (randNum <= boundP){
            Point p;
            p.x = 0;
            p.y = row;
            vertices.push_back(p);
            trueNumVertices++;
        }
        randNum = ((double) rand() / (RAND_MAX));
        if (randNum <= boundP){
            Point p;
            p.x = cols-1;
            p.y = row;
            vertices.push_back(p);
            trueNumVertices++;
        }
    }
    for (int col = 1; col < cols-1; col++)
    {
        double randNum = ((double) rand() / (RAND_MAX));
        if (randNum <= boundP){
            Point p;
            p.x = col;
            p.y = 0;
            vertices.push_back(p);
            trueNumVertices++;
        }
        randNum = ((double) rand() / (RAND_MAX));
        if (randNum <= boundP){
            Point p;
            p.x = col;
            p.y = rows-1;
            vertices.push_back(p);
            trueNumVertices++;
        }
    }

    // inner part
    for (int row = 1; row < rows-1; row++)
    {
        for (int col = 1; col < cols-1; col++)
        {
            double randNum = ((double) rand() / (RAND_MAX));
            if (grad.at<float>(row, col) > edgeThresh)
            {
                if (randNum <= edgeP)
                {
                    Point p;
                    p.x = col;
                    p.y = row;
                    vertices.push_back(p);
                    trueNumVertices++;
                }
            } else
            {
                if (randNum <= nonEdgeP)
                {
                    Point p;
                    p.x = col;
                    p.y = row;
                    vertices.push_back(p);
                    trueNumVertices++;
                }
            }
        }
    }

    numVertices = trueNumVertices;

    return vertices;
}

cv::Mat drawVoronoi(vector<int>& owner, int rows, int cols, int numVertices)
{
    cv::Mat voronoi = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(0,0,0));
    vector<cv::Vec3b> rgbMap(numVertices);

    for (int i = 0; i < numVertices; i++)
    {
        rgbMap[i][0] = rand() % 256;
        rgbMap[i][1] = rand() % 256;
        rgbMap[i][2] = rand() % 256;
    }

    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            voronoi.at<cv::Vec3b>(i, j) = rgbMap[owner[i * cols + j]];
        }
    }
    return voronoi;
}

// code from https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
int sign (Point p1, Point p2, Point p3)
{
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

bool PointInTriangle (Point pt, Point v1, Point v2, Point v3)
{
    int d1, d2, d3;
    bool has_neg, has_pos;

    d1 = sign(pt, v1, v2);
    d2 = sign(pt, v2, v3);
    d3 = sign(pt, v3, v1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}

cv::Mat drawTriangle(vector<Triangle>& triangles, cv::Mat& img)
{
    int rows = img.rows;
    int cols = img.cols;
    cv::Mat triImg = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(0,0,0));
    for (Triangle tri: triangles)
    {
        Point p = tri.center();
        int minX = min(tri.points[0].x, tri.points[1].x);
        minX = min(minX, tri.points[2].x);
        int minY = min(tri.points[0].y, tri.points[1].y);
        minY = min(minY, tri.points[2].y);    
        int maxX = max(tri.points[0].x, tri.points[1].x);
        maxX = max(maxX, tri.points[2].x);
        int maxY = max(tri.points[0].y, tri.points[1].y);
        maxY = max(maxY, tri.points[2].y);  
        for(int i=minY; i<=maxY; i++)
        {
            for(int j=minX; j<=maxX; j++)
            {
                Point pt;
                pt.x = j;
                pt.y = i;
                if (PointInTriangle(pt, tri.points[0], tri.points[1], tri.points[2]))     
                    triImg.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(p.y, p.x);                    
            }
        }        
    }
    return triImg;
}

cv::Mat drawTriangleLineOnImg(vector<Triangle>& triangles, cv::Mat& img)
{  
    cv::Mat triLine = img.clone();
    for (Triangle tri: triangles)
    {
        cv::Point p1, p2, p3;
        p1.x = tri.points[0].x;
        p1.y = tri.points[0].y;
        p2.x = tri.points[1].x;
        p2.y = tri.points[1].y;
        p3.x = tri.points[2].x;
        p3.y = tri.points[2].y;
        cv::line(triLine, p1, p2, cv::Scalar( 0, 0, 0));
        cv::line(triLine, p2, p3, cv::Scalar( 0, 0, 0));
        cv::line(triLine, p3, p1, cv::Scalar( 0, 0, 0));
    }
    return triLine;
}