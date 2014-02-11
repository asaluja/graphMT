#include "graph.h"
#include <omp.h>

using namespace std;
using namespace Eigen;

Graph::Graph(FeatureExtractor* features){
  sim_mat = SparseMatrix<double>();
  //use inverted index and similarities to compute similarities in parallel fashion
}

Graph::~Graph(){
}

void Graph::writeToFile(const string simMatLoc){
}
