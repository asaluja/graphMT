#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "featext.h"

using namespace std;
using namespace Eigen;

class Graph{
 public:
  Graph(FeatureExtractor* features);
  ~Graph();
  void writeToFile(const string simMatLoc);

 private:
  SparseMatrix<double> sim_mat; 
};
