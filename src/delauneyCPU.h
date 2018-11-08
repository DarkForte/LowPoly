#pragma once

#include <vector>
#include "point.h"

using namespace std;

int NextPower2(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

inline bool InBound(Point p, int row, int col)
{
    return (0 <= p.x && p.x < col && 0<=p.y && p.y < row);
}

inline int Index(Point p, int col)
{
    return p.y * col + p.x;
}

void DelauneyCPU(vector<Point> &seeds, vector<int> &owner, int rows, int cols)
{
    int start_stepsize = NextPower2(min(rows, cols)) / 2;

    const Point dir[] = {Point(0,1), Point(0, -1), Point(1, 0), Point(-1, 0),
                               Point(1,1), Point(1, -1), Point(-1, 1), Point(-1,-1)};

    for(int i=0; i<seeds.size(); i++)
    {
        Point seed = seeds[i];
        owner[Index(seed, cols)] = i;
    }

    // 1. Jump-Flooding
    for(int stepsize = start_stepsize; stepsize>=1; stepsize /= 2)
    {
        for(int r = 0; r < rows; r++)
        {
            for(int c = 0; c < cols; c++)
            {
                Point now_point(c, r);
                for(Point now_dir: dir)
                {
                    Point now_looking = now_point + now_dir * stepsize;
                    if(!InBound(now_looking, rows, cols))
                        continue;

                    if(owner[Index(now_looking, cols)] == -1)
                        continue;

                    int cand_dist = dist(seeds[owner[Index(now_looking, cols)]], now_point);
                    int now_owner = owner[Index(now_point, cols)];

                    if(now_owner == -1 || cand_dist < dist(seeds[now_owner], now_point))
                        owner[Index(now_point, cols)] = owner[Index(now_looking, cols)];
                }
            }
        }
    }

    const Point neighbor_dir[] = {Point(1, 0), Point(0, 1), Point(1, 1)};

}