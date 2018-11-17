#include "triangle.h"

Triangle::Triangle(Point a, Point b, Point c)
{
    points[0] = a;
    points[1] = b;
    points[2] = c;
}

Point Triangle::center()
{
    return (points[0] + points[1] + points[2]) / 3;
}
