#pragma once

#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <algorithm>

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
      count = 0; 
    }
    ~Phrase(){
    }
    bool isLabeled(){ 
      return labeled; 
    }
    const int id;
    const string phrase_str;
    const bool labeled;
    int count; 
    map<int,double> label_distribution; 
  };

  void addLabeledPhrasesFromFile(const string filename, const unsigned int PL, const string format);  
  void addUnlabeledPhrasesFromFile(const string filename, const unsigned int PL, const string out_filename, const bool analyze); 
  void printLabels(const string phrase);
  void normalizeLabelDistributions();
  int readMBestListFromFile(const string filename_in, const string filename_out, const vector<Phrase*> unlabeled_phrases); 
  void addGeneratedPhrases(const vector<string> generated_phrases); 
  void readPhraseIDsFromFile(const string filename, const bool readLabeled); 
  void writePhraseIDsToFile(const string filename, const bool writeLabeled); 
  Phrase* getNthPhrase(const unsigned int N){ return all_phrases[N]; }
  unsigned int getNumUnlabeledPhrases() { return numUnlabeled; }  
  unsigned int getNumLabeledPhrases() { return numLabeled; }
  int getPhraseID(const string phraseStr){ return (phrStr2ID.find(phraseStr) == phrStr2ID.end()) ? -1 : phrStr2ID[phraseStr]; }
  vector<Phrase*> getUnlabeledPhrases(){
    vector<Phrase*> unlabeled_phrases(numUnlabeled);
    copy_if(all_phrases.begin(), all_phrases.end(), unlabeled_phrases.begin(), [](Phrase* phrase) { return !phrase->isLabeled(); }); 
    return unlabeled_phrases; 
  }

 private:
  void initPhraseFromFile(string line, const unsigned int phrase_length, const string format);
  Phrase* initPhrase(const string srcPhr, const vector<string> srcTokens, bool isLabeled);
  void addLabelMoses(Phrase* phrase, vector<string> elements);
  void addLabelCdec(Phrase* phrase, vector<string> elements); 
  vector<string> multiCharSplitter(string line); 
  void analyzeUnlabeledPhrases(map<const string, unsigned int>& ngram_count); 

  vector<Phrase*> all_phrases;
  map<string, unsigned int> phrStr2ID;
  map<string, unsigned int> label_phrStr2ID;
  map<string, unsigned int> vocab; 
  phrasePair_Length max_tgtPL; 
  unsigned int numLabeled;
  unsigned int numUnlabeled; 
};
