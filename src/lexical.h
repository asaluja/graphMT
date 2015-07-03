#pragma once

#include <string>
#include <vector>
#include "extractor/translation_table.h"

using namespace std;

class LexicalScorer {
 public:
  LexicalScorer(const string location);
  vector<pair<double,double> > scorePhrasePairs(const vector<string> srcPhrases, const vector<string> tgtPhrases); 
  
 private:
  string lexModel_loc; 
  extractor::TranslationTable table; 
};
