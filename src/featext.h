#pragma once

#include <string>
#include <vector>
#include <tuple>

using namespace std;
typedef tuple<string,int,int> ngram_triple;

class Features {
 public:
  Features();
  ~Features();
  static vector<ngram_triple> extractNGrams(int n, string str);
  
 private:
  static string concat(vector<string> words, int start, int end); 
};
