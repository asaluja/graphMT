#include "featext.h"
#include <string>
#include <vector>
#include <tuple>
#include <boost/algorithm/string.hpp>

using namespace std;

vector<ngram_triple> Features::extractNGrams(int n, string str){
  vector<ngram_triple> ngrams = vector<ngram_triple>();
  vector<string> words; 
  boost::split(words, str, boost::is_any_of(" ")); //tokenizes the input sentence
  for (unsigned int i = 0; i < words.size() - n + 1; i++ ){
    ngram_triple triple = make_tuple(concat(words, i, i+n), i, i + n - 1); 
    ngrams.push_back(triple); 
  }
  return ngrams;
}

string Features::concat(vector<string> words, int start, int end){
  string ngram = "";
  for (int i = start; i < end; i++)
    ngram += ((i > start) ? " " : "") + words[i]; 
  return ngram; 
}
