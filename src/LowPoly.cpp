// LowPoly.cpp : Defines the entry point for the application.
//

#include <vector>
#include "point.h"
#include "LowPoly.h"

using namespace std;

int main()
{
  int number_seeds;
  cin>>number_seeds;
  
  int rows, cols;
  cin>>rows>>cols;

  vector<Point> seeds(number_seeds);
  vector<int> owner(rows * cols, -1);

  for (int i = 0; i < number_seeds; i++)
  {
    seeds[i].x = rand() % rows;
    seeds[i].y = rand() % cols;
  }
  
	return 0;
}
