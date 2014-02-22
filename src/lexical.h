#pragma once

#include <string>
#include <vector>

using namespace std;

class LexicalScorer {
 public:
  LexicalScorer(const string location);
  ~LexicalScorer();
  vector<pair<double,double> > scorePhrasePairs(vector<string> srcPhrases, vector<string> tgtPhrases);
  
 private:
  string lexModel_loc; 
    
};
