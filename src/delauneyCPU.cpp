#include <vector>
#include <unordered_set>
#include "point.h"
#include "triangle.h"
#include "delauney.h"

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

vector<Triangle> DelauneyCPU(vector<Point> &seeds, vector<int> &owner, int rows, int cols)
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

    vector<Triangle> triangles;
    const Point neighbor_dir[] = {Point(1, 0), Point(0, 1), Point(1, 1)};
    for(int r=0; r < rows - 1; r++)
    {
        for(int c=0; c<cols -1 ; c++)
        {
            Point now_point(c, r);
            unordered_set<int> colors;
            colors.insert(owner[Index(now_point, cols)]);
            for(Point now_dir: neighbor_dir)
            {
                Point next_point = now_point + now_dir;
                colors.insert(owner[Index(next_point, cols)]);
            }

            if(colors.size() == 3)
            {
                Triangle triangle;
                int p = 0;
                for(int index: colors)
                    triangle.points[p++] = seeds[index];

                triangles.push_back(triangle);
            }
            else if(colors.size() == 4)
            {
                Triangle triangle1(seeds[owner[Index(now_point, cols)]],
                                   seeds[owner[Index(now_point + Point(1, 0), cols)]],
                                   seeds[owner[Index(now_point + Point(0, 1), cols)]]);

                Triangle triangle2(seeds[owner[Index(now_point + Point(1, 0), cols)]],
                                   seeds[owner[Index(now_point + Point(0, 1), cols)]],
                                   seeds[owner[Index(now_point + Point(1, 1), cols)]]);

                triangles.push_back(triangle1);
                triangles.push_back(triangle2);
            }
        }
    }

    return triangles;
}