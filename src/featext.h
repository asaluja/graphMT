#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <set>
#include <map>
#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include "phrases.h"

using namespace std;
using namespace Eigen; 
typedef tuple<string,unsigned int,unsigned int> ngram_triple;
//typedef unordered_map<unsigned int, set<unsigned int> >::iterator invIdxIter;
typedef map<unsigned int, set<unsigned int> >::iterator invIdxIter;
typedef Triplet<double> triplet; 

class FeatureExtractor {
 public:
  FeatureExtractor();
  ~FeatureExtractor();
  static vector<ngram_triple> extractNGrams(const unsigned int n, const string str);
  vector<string> filterSentences(const string mono_dir_loc, Phrases* phrases, const unsigned int minPL, const unsigned int maxPL, const unsigned int maxPhrCount, const string monolingual_out); 
  void readStopWords(const string filename, const unsigned int num_sw); 
  void extractFeatures(Phrases* phrases, const string mono_filename, const unsigned int winsize, const unsigned int minPL, const unsigned int maxPL); 
  void pruneFeaturesByCount(const unsigned int minCount); 
  void analyzeFeatureMatrix(const vector<Phrases::Phrase*> unlabeled_phrases); 
  void rescaleCoocToPMI();
  void writeToFile(const string featMatLoc, const string invIdxLoc); 
  
 private:
  enum ContextSide { Left, Right };  
  static string concat(vector<string> words, const unsigned int start, const unsigned int end); 
  void addContext(const unsigned int phraseID, vector<string> subsent, const ContextSide side); 
  unsigned int getSetFeatureID(string featStr, const ContextSide side);
  set<unsigned int> stop_words; 
  map<string, unsigned int> featStr2ID; 
  //unordered_map<unsigned int, set<unsigned int> > inverted_idx; //right now a map between feature IDs and sets of phrase IDs, but can also make it a map to vector of Phrase* pointers? 
  map<unsigned int, set<unsigned int> > inverted_idx; 
  vector<triplet> featMat_triplets; 
  SparseMatrix<double,RowMajor> feature_matrix; 
};
