// LowPoly.cpp : Defines the entry point for the application.
//
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "point.h"
#include "LowPoly.h"
#include "delauneyCPU.h"
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


vector<Point> selectVertices(cv::Mat &grad, float edgeThresh, float edgeP, float nonEdgeP, int &numVertices)
{
    int rows = grad.rows;
    int cols = grad.cols;
    vector<Point> vertices;
    int trueNumVertices = 0;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
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

vector<Point> InputFromFile(char* filePath, int &numVertices, int &rows, int &cols)
{
    freopen(filePath, "r", stdin);
    vector<Point> ret;
    cin>>numVertices>>rows>>cols;
    int i;
    for(int i=1; i<=numVertices; i++)
    {
        int x, y;
        cin>>x>>y;
        ret.emplace_back(x, y);
    }
    return ret;
}

vector<Point> InputFromImage(char* imgPath, int numVertices, int &rows, int &cols, cv::Mat& img, float edgePortion, float edgeThresh)
{
    // Read image, set rows and cols
    img = cv::imread(imgPath);
    if (!img.data)
    {
        printf("Error loading image %s\n", imgPath);
        exit(-1);
    }
    rows = img.rows;
    cols = img.cols;
    int numPixel = rows * cols;

    // get grad (edge)
    cv::Mat grad = getGrad(img);

    // calculate threshold for selecting edge pixel and non-edge Pixel
    int numEdgePix = 0;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            if (grad.at<float>(row, col) >= edgeThresh)
                numEdgePix++;
        }
    }
    int numEdgeV = min((int) (numVertices * edgePortion), numEdgePix);
    int numNonEdgeV = numVertices - numEdgeV;
    float edgeP = (float) numEdgeV / numEdgePix;
    float nonEdgeP = (float) numNonEdgeV / (numPixel - numEdgePix);

    // select points on image
    vector<Point> vertices = selectVertices(grad, edgeThresh, edgeP, nonEdgeP, numVertices);

    // write the edge detection result and selected points to img
    cv::Mat pts = cv::Mat(rows, cols, CV_32F, 0.0);
    for (int i = 0; i < numVertices; i++)
    {
        int row = vertices[i].y;
        int col = vertices[i].x;
        pts.at<float>(row, col) = 255.0;
    }
    cv::imwrite("points.png", pts);
    cv::Mat edges = grad * 255.0;
    cv::imwrite("edges.png", edges);

    return vertices;
}

void drawVoronoi(vector<int>& owner, int rows, int cols, int numVertices)
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
    // write to image
    cv::imwrite("voronoi.png", voronoi);
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

void drawTriangle(vector<Triangle>& triangles, cv::Mat& img)
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
        minY = min(minX, tri.points[2].y);    
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
    // wirte to image
    cv::imwrite("triangle.png", triImg);
}

int main(int argc, char **argv)
{
    vector<Point> vertices;
    int rows, cols;

    char *imgPath;
    cv::Mat img;
    int numVertices = 500;
    float edgePortion = 0.8; // percentage of points being on edge
    float edgeThresh = 0.1; // threshold for a point being an edge
    // parse inputs
    int opt;
    while ((opt = getopt(argc, argv, "f:i:v:e:")) != -1)
    {
        if(opt == 'f')
        {
            vertices = InputFromFile(optarg, numVertices, rows, cols);
        }
        else
        {
            switch (opt)
            {
                case 'i':
                    imgPath = optarg;
                    break;
                case 'v':
                    numVertices = atoi(optarg);
                    break;
                case 'e':
                    edgePortion = atof(optarg);
                    break;
                default:
                    cout << "Unrecognized argument: " << opt;
            }

            vertices = InputFromImage(imgPath, numVertices, rows, cols, img, edgePortion, edgeThresh);
        }
    }

    vector<int> owner(rows * cols, -1);

    vector<Triangle> triangles = DelauneyCPU(vertices, owner, rows, cols);

    drawVoronoi(owner, rows, cols, numVertices);

    drawTriangle(triangles, img);

    // for(int i=0; i<rows; i++)
    // {
    //     for(int j=0; j<cols; j++)
    //     {
    //         cout<<owner[i * cols + j]<<" ";
    //     }
    //     cout<<endl;
    // }

    // for(Triangle triangle: triangles)
    // {
    //     for(int i=0; i<=2; i++)
    //     {
    //         cout<<triangle.points[i].x <<" "<<triangle.points[i].y<<"; ";
    //     }
    //     cout<<endl;
    // }

    return 0;
}
