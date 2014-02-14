#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include "featext.h"

using namespace std;
using namespace Eigen;
typedef Triplet<double> triplet;

class Graph{
 public:
  Graph(FeatureExtractor* features, const unsigned int k);
  ~Graph();
  void writeToFile(const string simMatLoc);
  void analyzeSimilarityMatrix(const vector<Phrases::Phrase*> unlabeled_phrases); 

 private:
  SparseMatrix<double> sim_mat; 
  vector<triplet> sim_mat_triplets; 
};
