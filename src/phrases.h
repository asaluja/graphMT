#pragma once

#include <string>
#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include "lexical.h"

using namespace std;
typedef tuple<string, string, unsigned int> phrasePair_Length;

class Phrases {
 public:
  Phrases();
  explicit Phrases(const Phrases* orig_phrases);
  ~Phrases();
  struct Phrase {    
  Phrase(const int phrID, const string phrStr, const bool label) : 
    id(phrID), phrase_str(phrStr), labeled(label) {
      label_distribution = map<int, double>();
      marginal = 0; 
    }
    ~Phrase(){
    }
    bool isLabeled(){ 
      return labeled; 
    }
    set<int> getLabels(){
      set<int> labels = set<int>();
      for (map<int,double>::iterator it = label_distribution.begin(); it != label_distribution.end(); it++)
	labels.insert(it->first); 
      return labels; 
    }
    const int id;
    const string phrase_str;
    const bool labeled;
    double marginal; 
    map<int,double> label_distribution; 
    void normalizeDistribution(){
      typedef map<int,double>::iterator iter; 
      double normalizer = 0.0; 
      for (iter it = label_distribution.begin(); it != label_distribution.end(); it++ )
	normalizer += it->second;
      for (iter it = label_distribution.begin(); it != label_distribution.end(); it++ )
	it->second /= normalizer;
    }
  };

  void addLabeledPhrasesFromFile(const string filename, const unsigned int PL, const string format);  
  void addUnlabeledPhrasesFromFile(const string filename, const unsigned int PL, const string out_filename, const bool analyze); 
  void printLabels(const string phrase);
  void normalizeLabelDistributions();
  int readMBestListFromFile(const string filename_in, const string filename_out, const vector<Phrase*> unlabeled_phrases); 
  static map<const string, vector<string> > readFormattedMBestListFromFile(const string mbest_processed_loc);   
  void addGeneratedPhrases(const vector<string> generated_phrases); 
  void readPhraseIDsFromFile(const string filename, const bool readLabeled);   
  void writePhraseIDsToFile(const string filename, const bool writeLabeled); 
  void readLabelPhraseIDsFromFile(const string filename); 
  Phrase* getNthPhrase(const unsigned int N){ return all_phrases[N]; }
  unsigned int getNumUnlabeledPhrases() { return numUnlabeled; }  
  unsigned int getNumLabeledPhrases() { return numLabeled; }
  unsigned int getPhraseID(const string phraseStr){ return (phrStr2ID.find(phraseStr) == phrStr2ID.end()) ? -1 : phrStr2ID[phraseStr]; }
  unsigned int getLabelPhraseID(const string labelPhraseStr){ return (label_phrStr2ID.find(labelPhraseStr) == label_phrStr2ID.end()) ? -1 : label_phrStr2ID[labelPhraseStr]; }
  string getLabelPhraseStr(const unsigned int labelPhraseID) { return (label_phrID2Str.find(labelPhraseID) == label_phrID2Str.end()) ? "" : label_phrID2Str[labelPhraseID]; }
  vector<Phrase*> getUnlabeledPhrases(){
    vector<Phrase*> unlabeled_phrases(numUnlabeled);
    copy_if(all_phrases.begin(), all_phrases.end(), unlabeled_phrases.begin(), [](Phrase* phrase) { return !phrase->isLabeled(); }); 
    return unlabeled_phrases; 
  }
  vector<Phrase*> getLabeledPhrases(){
    vector<Phrase*> labeled_phrases(numLabeled); 
    copy_if(all_phrases.begin(), all_phrases.end(), labeled_phrases.begin(), [](Phrase* phrase) {return phrase->isLabeled(); }); 
    return labeled_phrases; 
  }
  void computeMarginals(const string cooc_loc); 
  
  void writePhraseTable(Phrases* tgt_phrases, const string pt_format, const string new_pt_loc, LexicalScorer* const lex); 

 private:
  void initPhraseFromFile(string line, const unsigned int phrase_length, const string format);
  Phrase* initPhrase(const string srcPhr, const vector<string> srcTokens, const int phrID, bool isLabeled);
  void addLabelMoses(Phrase* phrase, vector<string> elements);
  void addLabelCdec(Phrase* phrase, vector<string> elements); 
  vector<string> multiCharSplitter(string line); 
  void analyzeUnlabeledPhrases(map<const string, unsigned int>& ngram_count); 

  vector<Phrase*> all_phrases;
  map<string, unsigned int> phrStr2ID;
  map<string, unsigned int> label_phrStr2ID;
  map<unsigned int, string> label_phrID2Str; 
  map<string, unsigned int> vocab; 
  phrasePair_Length max_tgtPL; 
  unsigned int numLabeled;
  unsigned int numUnlabeled; 
};
