#include "phrases.h"
#include "featext.h"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

namespace io = boost::iostreams;
namespace fs = boost::filesystem;
using namespace std;

const string delimiter = " ||| "; 

Phrases::Phrases(){
  phrPtrs = vector<Phrase*>();
  phrStr2ID = map<string, int>();
  label_phrStr2ID = map<string, int>();
}

Phrases::~Phrases(){
  for (unsigned int i = 0; i < phrPtrs.size(); i++ )
    delete phrPtrs[i];
}

void Phrases::normalizeLabelDistributions(){  
  typedef map<int,double>::iterator iter;
  for (unsigned int i = 0; i < phrPtrs.size(); i++ ){
    map<int,double>* labels = &phrPtrs[i]->label_distribution; 
    double normalizer = 0;
    for (iter it = labels->begin(); it != labels->end(); it++ )
      normalizer += it->second;
    for (iter it = labels->begin(); it != labels->end(); it++ )
      it->second /= normalizer;
  }
}

void Phrases::printLabels(string phrase){
  int phrID = phrStr2ID[phrase];
  map<int,double> labels = phrPtrs[phrID]->label_distribution; 
  typedef map<int,double>::iterator it;
  for (it i = labels.begin(); i != labels.end(); i++)
    cout << "Key: " << i->first << "; Value: " << i->second << endl; 
}

void Phrases::addUnlabeledPhrasesFromFile(string filename, int PL){
  map<string, int> ngram_count = map<string, int>();  
  ifstream eval_corpus; 
  eval_corpus.exceptions(ios::failbit | ios::badbit);
  eval_corpus.open(filename.c_str(), ios_base::in | ios_base::binary);
  for (string line; getline(eval_corpus, line);){
    boost::trim(line); 
    vector<ngram_triple> ngrams_from_line = Features::extractNGrams(PL, line); 
    string ngram;
    for (unsigned int i = 0; i < ngrams_from_line.size(); i++ ){
      tie(ngram, ignore, ignore) = ngrams_from_line[i]; 
      pair<map<string, int>::iterator,bool> ret; 
      ret = ngram_count.insert(pair<string, int>(ngram, 1)); 
      if (ret.second == false)
	ngram_count[ngram]++;
    }
  }
  cout << "Number of " << PL << "-grams extracted from evaluation corpus: " << ngram_count.size() << endl;   
  //add unlabeled phrases to phrases
  //write out to file
  //take option for analyzeUnlabeled
}

//keep a track of max target side PL from phrase table
void Phrases::addLabeledPhrasesFromFile(string filename, int PL){
  ifstream pt_file; //file handle for phrase table
  pt_file.exceptions(ios::failbit | ios::badbit); 
  pt_file.open(filename.c_str(), ios_base::in | ios_base::binary);  
  fs::path p(filename); 
  int numPhrases = 0; 
  if (p.extension() == ".gz"){ //special handling for .gz files
    io::filtering_stream<io::input> decompressor;
    decompressor.push(io::gzip_decompressor());
    decompressor.push(pt_file);
    for (string line; getline(decompressor, line);){
      initPhraseFromFile(line, PL); 
      numPhrases++;
    }
  }
  else { //for non .gz files --> test this out properly!
    for (string line; getline(pt_file, line);){
      initPhraseFromFile(line, PL);
      numPhrases++;
    }
  }
  cout << "Number of phrases in phrase table: " << numPhrases << endl; 
  cout << "Number of phrases with desired phrase length " << PL << ": " << phrPtrs.size() << endl; 
}

void Phrases::initPhraseFromFile(string line, int phrase_length){
  boost::trim(line);
  vector<string> elements = multiCharSplitter(line); 
  string srcPhr = elements[0];
  vector<string> srcTokens;
  boost::split(srcTokens, srcPhr, boost::is_any_of(" "));
  if (srcTokens.size() == phrase_length){
    Phrase* phrase = NULL;
    if (phrStr2ID.find(srcPhr) == phrStr2ID.end()){ //new phrase
      int phrID = phrPtrs.size();
      phrase = new Phrase(phrID, srcPhr, true); 
      phrStr2ID[srcPhr] = phrID;
      phrPtrs.push_back(phrase);
    }
    else 
      phrase = phrPtrs[phrStr2ID[srcPhr]];
    addLabel(phrase, elements); 
  }
}

vector<string> Phrases::multiCharSplitter(string line){
  size_t pos = 0;
  vector<string> elements = vector<string>();
  string token; 
  while ((pos = line.find(delimiter)) != string::npos){
    token = line.substr(0, pos);
    elements.push_back(token);
    line.erase(0, pos + delimiter.length());
  }
  elements.push_back(line); 
  return elements;
}

//to add: keep track of maximum phrase length on target side, and which source/target pair it is
void::Phrases::addLabel(Phrase* phrase, vector<string> elements){
  string tgtPhr = elements[1];
  string featStr = elements[2];
  vector<string> featStrVec;
  boost::split(featStrVec, featStr, boost::is_any_of(" ")); 
  double fwdPhrProb = atof(featStrVec[0].c_str());
  int label_id; 
  if (label_phrStr2ID.find(tgtPhr) == label_phrStr2ID.end()){ //new label phrase
    label_id = label_phrStr2ID.size();
    label_phrStr2ID[tgtPhr] = label_id;
  }
  else
    label_id = label_phrStr2ID[tgtPhr];
  phrase->label_distribution[label_id] = fwdPhrProb;    
}
  
