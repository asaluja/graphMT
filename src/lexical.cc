#include "lexical.h"
#include <fstream>
#include <cmath>
#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include "extractor/translation_table.h"
#include "extractor/data_array.h"

using namespace std; 
const string NULL_WORD_STR = "__NULL__"; 
const int MAXSCORE = 99; 

LexicalScorer::LexicalScorer(const string location){
  lexModel_loc = location; 
  assert(access(location.c_str(), F_OK) != -1); //assert for presence of lex model
  table = extractor::TranslationTable();
  ifstream ttable_fstream(lexModel_loc); 
  boost::archive::binary_iarchive ttable_stream(ttable_fstream); 
  ttable_stream >> table;           
}

vector<pair<double,double> > LexicalScorer::scorePhrasePairs(const vector<string> srcPhrases, const vector<string> tgtPhrases){
  assert(srcPhrases.size() == tgtPhrases.size()); 
  vector<pair<double,double> > lex_scores = vector<pair<double,double> >(); 
  for (unsigned int i = 0; i < srcPhrases.size(); i++){
    vector<string> src_words, tgt_words; 
    boost::split(src_words, srcPhrases[i], boost::is_any_of(" ")); 
    boost::split(tgt_words, tgtPhrases[i], boost::is_any_of(" ")); 
    tgt_words.push_back(NULL_WORD_STR); 
    double bwd_score = 0; 
    for (unsigned int j = 0; j < src_words.size(); j++){ //starting lex(f|e) computation
      double max_score = 0; 
      for (unsigned int k = 0; k < tgt_words.size(); k++)
	max_score = max(max_score, table.GetSourceGivenTargetScore(src_words[j], tgt_words[k])); 
      bwd_score += max_score > 0 ? -log10(max_score) : MAXSCORE; 
    }
    src_words.push_back(NULL_WORD_STR); 
    tgt_words.pop_back(); //removes null word string from target words
    double fwd_score = 0; 
    for (unsigned int j = 0; j < tgt_words.size(); j++){ //starting lex(e|f) computation
      double max_score = 0;
      for (unsigned int k = 0; k < src_words.size(); k++)
	max_score = max(max_score, table.GetTargetGivenSourceScore(src_words[k], tgt_words[j])); 
      fwd_score += max_score > 0 ? -log10(max_score) : MAXSCORE; 
    }
    pair<double, double> fwd_bwd_lex = make_pair(pow(10, -fwd_score), pow(10, -bwd_score)); 
    lex_scores.push_back(fwd_bwd_lex); 
  }
  return lex_scores; 
}



