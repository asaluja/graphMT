#include "graph.h"
#include <omp.h>
#include <set>

using namespace std;
using namespace Eigen;

Graph::Graph(FeatureExtractor* features, const unsigned int k){
  sim_mat = SparseMatrix<double>();
  sim_mat_triplets = vector<triplet>(); 
  unsigned int featureless_phrases = 0; 
  unsigned int negative_similarities = 0; 
  #pragma omp parallel for
  for (unsigned int i = 0; i < features->getNumPoints(); i++){
    SparseVector<double> featureVec = features->getFeatureRow(i);     
    set<unsigned int> neighbors = set<unsigned int>();
    for (SparseVector<double>::InnerIterator it(featureVec); it; ++it){
      set<unsigned int> neighbors_by_feature = features->getNeighbors(it.index()); 
      neighbors.insert(neighbors_by_feature.begin(), neighbors_by_feature.end()); //or use set_union instead? which is faster? 
    }
    if (neighbors.size() > 0){
      vector<pair<unsigned int, double> > idxsAndDotProds = vector<pair<unsigned int, double> >(); 
      idxsAndDotProds.reserve(neighbors.size()); 
      set<unsigned int>::iterator iter; 
      for (iter = neighbors.begin(); iter != neighbors.end(); iter++){
	if ((*iter) != i){ //filtering for self similarity	  
	  double dp = featureVec.dot(features->getFeatureRow(*iter)) / (featureVec.norm() * features->getFeatureRow(*iter).norm()); 
	  if (dp > 0)
	    idxsAndDotProds.push_back(make_pair(*iter, dp)); 
	}
      }
      if (idxsAndDotProds.size() > 0){
	sort(idxsAndDotProds.begin(), idxsAndDotProds.end(), [](const pair<unsigned int, double>& lhs, const pair<unsigned int, double>& rhs){ return lhs.second > rhs.second; }); //in ascending order
	unsigned int topN = (k < idxsAndDotProds.size()) ? k : idxsAndDotProds.size(); 
	vector<pair<unsigned int, double> > topK_idxsDPs(idxsAndDotProds.begin(), idxsAndDotProds.begin()+topN);     
        #pragma omp critical(addSparseTriplet)
	{
	  for (unsigned int j = 0; j < topK_idxsDPs.size(); j++)
	    sim_mat_triplets.push_back(triplet(i, topK_idxsDPs[j].first, topK_idxsDPs[j].second)); 
	  sim_mat_triplets.push_back(triplet(i, i, 1.0)); 
	}
      }
      else { 
	negative_similarities++; 
        #pragma omp critical(addSparseTripletSelfSim)
	{
	  sim_mat_triplets.push_back(triplet(i, i, 1.0)); 
	}
      }
    }
    else { 
      featureless_phrases++; 
      #pragma omp critical(addSparseTripletSelfSimFeatureless)
	{
	  sim_mat_triplets.push_back(triplet(i, i, 1.0)); 
	}
    }    
  }
  cout << "Number of phrases without neighbors (i.e., other phrases sharing one common non stop-word feature): " << featureless_phrases << endl; 
  cout << "Number of phrases that have negative similarities with all neighbors: " << negative_similarities << endl; 
  sim_mat.resize(features->getNumPoints(), features->getNumPoints()); 
  sim_mat.reserve(sim_mat_triplets.size()); 
  sim_mat.setFromTriplets(sim_mat_triplets.begin(), sim_mat_triplets.end()); 
  sim_mat_triplets.clear(); //memory efficiency purposes
  cout << "Before symmetrizing, total NNZs in similarity matrix: " << sim_mat.nonZeros() << endl; 
  sim_mat = 0.5*(SparseMatrix<double>(sim_mat.transpose()) + sim_mat); 
  VectorXd indSimSumRowInv = (sim_mat*VectorXd::Ones(sim_mat.cols())).cwiseInverse(); 
  SparseMatrix<double,RowMajor> left_mult(sim_mat.rows(), sim_mat.rows());
  vector<triplet> left_mult_diagonal = vector<triplet>();
  for (unsigned int i = 0; i < indSimSumRowInv.size(); i++)
    left_mult_diagonal.push_back(triplet(i, i, indSimSumRowInv[i])); 
  left_mult.reserve(left_mult_diagonal.size());
  left_mult.setFromTriplets(left_mult_diagonal.begin(), left_mult_diagonal.end()); 
  sim_mat = left_mult * sim_mat; 
  cout << "After symmetrizing (and normalizing), total NNZs in random walk matrix: " << sim_mat.nonZeros() << endl; 
}

Graph::~Graph(){
}

void Graph::writeToFile(const string simMatLoc){
  saveMarket(sim_mat, simMatLoc); 
}

void Graph::analyzeSimilarityMatrix(const vector<Phrases::Phrase*> unlabeled_phrases){
  set<int> unlabeled_ids = set<int>();
  for (unsigned int i = 0; i < unlabeled_phrases.size(); i++)
    unlabeled_ids.insert(unlabeled_phrases[i]->id); 
  cout << "Dimensions: " << sim_mat.rows() << " x " << sim_mat.cols() << "; NNZs: " << sim_mat.nonZeros() << endl;
  vector<unsigned int> row_lengths = vector<unsigned int>();
  row_lengths.reserve(sim_mat.rows()); 
  int zr_lab = 0, zr_unl = 0; 
  for (int i = 0; i < sim_mat.rows(); i++){
    if (sim_mat.row(i).nonZeros() == 1){ //every phrase has a self sim      
      if (unlabeled_ids.find(i) == unlabeled_ids.end())
	zr_lab++;
      else
	zr_unl++;
    }
    row_lengths.push_back(sim_mat.row(i).nonZeros()); 
  }
  vector<unsigned int>::iterator iter = max_element(row_lengths.begin(), row_lengths.end()); 
  unsigned int idx = distance(row_lengths.begin(), iter); 
  cout << "Phrase ID " << idx << " has the most neighbors: " << *iter << endl; 
  cout << "Number of completely disconnected nodes: " << zr_lab + zr_unl << endl; 
  cout << "Number of completely disconnected labeled nodes: " << zr_lab << endl; 
  cout << "Number of completely disconnected unlabeled nodes: " << zr_unl << endl; 
}
