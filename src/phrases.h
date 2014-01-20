#pragma once

#include <string>
#include <vector>
#include <map>
#include <tuple>

using namespace std;
typedef tuple<string,string,int> phrasePair_Length;

class Phrases {
 public:
  Phrases();
  explicit Phrases(Phrases* orig_phrases);
  ~Phrases();
  struct Phrase {    
  Phrase(int phrID, string phrStr, bool label) : 
    id(phrID), phrase_str(phrStr), labeled(label) {
      label_distribution = map<int, double>();
    }
    ~Phrase(){
    }
    bool isLabeled(){ 
      return labeled; 
    }
    int id;
    string phrase_str;
    bool labeled;
    map<int,double> label_distribution; 
  };

  void addLabeledPhrasesFromFile(string filename, int PL);  
  void printLabels(string phrase);
  void normalizeLabelDistributions();
  void addUnlabeledPhrasesFromFile(string filename, int PL, string out_filename, bool analyze); 

 private:
  void initPhraseFromFile(string line, int phrase_length); 
  void addLabel(Phrase* phrase, vector<string> elements);
  vector<string> multiCharSplitter(string line); 
  void analyzeUnlabeledPhrases(map<string, int>& ngram_count); 

  vector<Phrase*> all_phrases;
  map<string, int> phrStr2ID;
  map<string, int> label_phrStr2ID;
  map<string, int> vocab; 
  phrasePair_Length max_tgtPL; 
  int numLabeled;
  int numUnlabeled; 
};
