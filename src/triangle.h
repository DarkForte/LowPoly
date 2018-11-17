#pragma once

#include "point.h"

struct Triangle
{
    Point points[3];
    Triangle(){}
    Triangle(Point a, Point b, Point c);

    Point center();

};
