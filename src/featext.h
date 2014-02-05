#pragma once

#include <string>
#include <vector>
#include <tuple>
#include "phrases.h"

using namespace std;
typedef tuple<string,int,int> ngram_triple;

class FeatureExtractor {
 public:
  FeatureExtractor();
  ~FeatureExtractor();
  static vector<ngram_triple> extractNGrams(const unsigned int n, const string str);
  vector<string> filterSentences(const string mono_dir_loc, Phrases* phrases, const unsigned int minPL, const unsigned int maxPL, const unsigned int maxPhrCount, const string monolingual_out); 
  
 private:
  static string concat(vector<string> words, const unsigned int start, const unsigned int end); 
};
