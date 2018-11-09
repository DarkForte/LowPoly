#pragma once

#include "point.h"

struct Triangle
{
    Point points[3];
    Triangle(){}
    Triangle(Point a, Point b, Point c)
    {
        points[0] = a;
        points[1] = b;
        points[2] = c;
    }

    Point center()
    {
        return (points[0] + points[1] + points[2]) / 3;
    }
};
