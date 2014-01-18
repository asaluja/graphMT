#pragma once

#include <string>
#include <vector>
#include <map>

using namespace std;

class Phrases {
 public:
  Phrases();
  ~Phrases();
  void addLabeledPhrasesFromFile(string filename, int PL);  
  void printLabels(string phrase);
  void normalizeLabelDistributions();
  void addUnlabeledPhrasesFromFile(string filename, int PL); 

  struct Phrase {    
  Phrase(int phrID, string phrStr, bool labeled) : 
    id(phrID), phrase_str(phrStr), isLabeled(labeled) {
      label_distribution = map<int, double>();
    }

    ~Phrase(){
    }

    int id;
    string phrase_str;
    bool isLabeled;
    map<int,double> label_distribution; 
  };

 private:
  void initPhraseFromFile(string line, int phrase_length); 
  void addLabel(Phrase* phrase, vector<string> elements);
  vector<string> multiCharSplitter(string line); 

  vector<Phrase*> phrPtrs; 
  map<string, int> phrStr2ID;
  map<string, int> label_phrStr2ID;
  
};
