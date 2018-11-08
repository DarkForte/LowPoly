struct Point
{
    int x, y;
    Point(){}
    Point(int _x, int _y):x(_x), y(_y){};
};

int dist(const Point &a, const Point &b)
{
    int dx = a.x - b.x;
    int dy = a.y - b.y;
    return dx*dx + dy*dy;  
}

Point operator + (const Point &a, const Point &b)
{
    return Point(a.x + b.x, a.y + b.y);
}

