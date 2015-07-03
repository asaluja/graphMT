#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include "featext.h"
#include "lexical.h"

using namespace std;
using namespace Eigen;
typedef Triplet<double> triplet;

class Graph{
 public:
  Graph(FeatureExtractor* features, const unsigned int k);
  explicit Graph(const string simMatLoc); 
  ~Graph();
  void writeToFile(const string simMatLoc);
  void analyzeSimilarityMatrix(const vector<Phrases::Phrase*> unlabeled_phrases); 
  void initLabelsWithLexScore(Phrases* src_phrases, const string mbest_processed_loc, LexicalScorer* const lex, const int maxCand_size, const bool filter_sw, set<int> stopWords=set<int>()); 
  void labelProp(Phrases* src_phrases); 
  void structLabelProp(Phrases* src_phrases, void* tgt_graph, bool dynamic_graph); //data is constant for tgt_graph, so we should put that
  double getSimilarity(const int i, const int j){ return sim_mat.coeff(i, j); }

 private:
  SparseMatrix<double,RowMajor> sim_mat; 
  vector<triplet> sim_mat_triplets; 
  map<int, double> generateCandidateTranslations(const string phrStr, const int phrID, Phrases* const src_phrases, const vector<string> mbest_candidates, LexicalScorer* const lex, const int maxCand_size, const bool filter_sw, set<int> stopWords=set<int>()); 
  void filterCandidatesForStopWords(set<int>& labels, const set<int> stopWords); 
};

class DynamicGraph{
 public:
  DynamicGraph(FeatureExtractor* features); 
  explicit DynamicGraph(const string dgLoc); 
  ~DynamicGraph();
  void writeToFile(const string dgLoc); 
  double getSimilarity(const int i, const int j); 
  int getCacheSize(){return cache.size(); } //for debugging
 private:
  map<string,double> cache; 
  SparseMatrix<double,RowMajor> feat_mat; 
};
