#pragma once

struct Point
{
    int x, y;
    Point(){}
    Point(int _x, int _y);
};


int dist(const Point &a, const Point &b);


Point operator + (const Point &a, const Point &b);

Point operator * (const Point &a, int b);

Point operator / (const Point &a, int b);