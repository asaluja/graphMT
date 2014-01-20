#include "phrases.h"
#include "featext.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include <set>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

namespace io = boost::iostreams;
namespace fs = boost::filesystem;
using namespace std;
const string delimiter = " ||| "; 

//standard constructor
Phrases::Phrases(){
  all_phrases = vector<Phrase*>();
  phrStr2ID = map<string, int>();
  label_phrStr2ID = map<string, int>();
  vocab = map<string, int>();
  max_tgtPL = make_tuple("", "", 0); 
  numLabeled = 0, numUnlabeled = 0; 
}

//constructor for initializing phrases of other side
Phrases::Phrases(Phrases* orig_phrases){
  all_phrases = vector<Phrase*>();
  phrStr2ID = map<string,int>(orig_phrases->label_phrStr2ID);
  label_phrStr2ID = map<string, int>();
  vocab = map<string, int>();
  max_tgtPL = make_tuple("", "", 0);
  numUnlabeled = phrStr2ID.size();
  numLabeled = 0;
  typedef map<string,int>::iterator iter;
  for (iter it = phrStr2ID.begin(); it != phrStr2ID.end(); it++){ //adding target phrases as Phrase structs
    Phrase* phrase = new Phrase(it->second, it->first, false); 
    all_phrases.push_back(phrase); 
    vector<string> tokens; 
    boost::split(tokens, it->first, boost::is_any_of(" "));    
    for (unsigned int i = 0; i < tokens.size(); i++){ //add unigrams to vocab if not seen before
      if (vocab.find(tokens[i]) == vocab.end()){ 
	int id = vocab.size();
	vocab[tokens[i]] = id;
      }
    }
  }
  cout << "Target vocabulary size: " << vocab.size() << endl; 
  cout << "Number of phrases: " << numLabeled + numUnlabeled << endl; 
}

Phrases::~Phrases(){
  for (unsigned int i = 0; i < all_phrases.size(); i++ )
    delete all_phrases[i];
}

//goes through label distribution for each labeled soure phrase and normalizes (sum = 1)
void Phrases::normalizeLabelDistributions(){  
  typedef map<int,double>::iterator iter;
  for (unsigned int i = 0; i < all_phrases.size(); i++ ){
    map<int,double>* labels = &all_phrases[i]->label_distribution; 
    double normalizer = 0;
    for (iter it = labels->begin(); it != labels->end(); it++ )
      normalizer += it->second;
    for (iter it = labels->begin(); it != labels->end(); it++ )
      it->second /= normalizer;
  }
}

//function primarily meant for debugging
void Phrases::printLabels(string phrase){
  int phrID = phrStr2ID[phrase];
  map<int,double> labels = all_phrases[phrID]->label_distribution; 
  typedef map<int,double>::iterator it;
  for (it i = labels.begin(); i != labels.end(); i++)
    cout << "Key: " << i->first << "; Value: " << i->second << endl; 
}

//goes through evaluation corpus, first extracts all n-grams of length PL, and then adds
//n-grams that aren't in phrase tablea s unlabeled n-grams. 
void Phrases::addUnlabeledPhrasesFromFile(string filename, int PL, string out_filename, bool analyze){
  map<string, int> ngram_count = map<string, int>();  
  ifstream eval_corpus; 
  eval_corpus.exceptions(ios::failbit | ios::badbit);
  eval_corpus.open(filename.c_str(), ios_base::in | ios_base::binary);
  string line;
  while (!eval_corpus.eof()){
    getline(eval_corpus, line); 
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
  eval_corpus.close();
  cout << "Number of " << PL << "-grams in evaluation corpus: " << ngram_count.size() << endl;   
  ofstream unlabeled_phrases;
  unlabeled_phrases.open(out_filename.c_str()); //write out unlabeled n-grams
  typedef map<string,int>::iterator iter;
  for (iter it = ngram_count.begin(); it != ngram_count.end(); it++ ){
    string srcPhr = it->first;
    iter checkUnlabeled = phrStr2ID.find(srcPhr);
    if (checkUnlabeled == phrStr2ID.end()){ //add phrases not in phrase table
      numUnlabeled++;
      Phrase* phrase = new Phrase(all_phrases.size(), srcPhr, false);     
      all_phrases.push_back(phrase); 
      unlabeled_phrases << srcPhr << endl; 
    }
  }
  cout << "Number of unlabeled " << PL << "-grams in evaluation corpus: " << numUnlabeled << endl; 
  unlabeled_phrases.close();
  if (analyze) //if defined, we can analyze the phrases
    analyzeUnlabeledPhrases(ngram_count);
}

//for analysis purposes: breaks down unknown n-grams into all known unigrams, some known, or none known.  
void Phrases::analyzeUnlabeledPhrases(map<string, int>& ngram_count){
  vector<Phrase*> unlabeled_phrases(numUnlabeled);
  copy_if(all_phrases.begin(), all_phrases.end(), unlabeled_phrases.begin(), [](Phrase* phrase) { return !phrase->isLabeled(); }); 
  int kk = 0, kkt = 0, uk = 0, ukt = 0, uu = 0, uut = 0;
  typedef map<string, int>::iterator iter;
  for (unsigned int i = 0; i < unlabeled_phrases.size(); i++){
    vector<string> srcTokens;
    boost::split(srcTokens, unlabeled_phrases[i]->phrase_str, boost::is_any_of(" "));
    set<bool> is_word_oov = set<bool>();
    for (unsigned int j = 0; j < srcTokens.size(); j++){
      iter checkOOV = vocab.find(srcTokens[j]);
      is_word_oov.insert((checkOOV == vocab.end()));
    }
    if (is_word_oov.size() > 1){ //mixed OOV and non-OOV
      uk++;
      ukt += ngram_count[unlabeled_phrases[i]->phrase_str];
    }
    else {
      set<bool>::iterator val = is_word_oov.begin();
      if (*val){ //unknown word
	uu++;
	uut += ngram_count[unlabeled_phrases[i]->phrase_str];
      }
      else {
	kk++;
	kkt += ngram_count[unlabeled_phrases[i]->phrase_str];
      }
    }
  }
  cout << "Out of these unlabeled phrases, " << kk << " of them consist of n-grams where all unigrams are known; this corresponds to " << kkt << " tokens." << endl; 
  cout << "Out of these unlabeled phrases, " << uk << " of them consist of n-grams where some unigrams are unknown; this corresponds to " << ukt << " tokens." << endl; 
  cout << "Out of these unlabeled phrases, " << uu << " of them consist of n-grams where all unigrams are unknown; this corresponds to " << uut << " tokens." << endl; 
}

//function that goes through phrase table file and initializes labeled phrases
void Phrases::addLabeledPhrasesFromFile(string filename, int PL){
  ifstream pt_file; //file handle for phrase table
  pt_file.exceptions(ios::failbit | ios::badbit); 
  pt_file.open(filename.c_str(), ios_base::in | ios_base::binary);  
  fs::path p(filename); 
  if (p.extension() == ".gz"){ //special handling for .gz files
    io::filtering_stream<io::input> decompressor;
    decompressor.push(io::gzip_decompressor());
    decompressor.push(pt_file);
    for (string line; getline(decompressor, line);){
      initPhraseFromFile(line, PL); 
      numLabeled++;
    }
  }
  else { //for non .gz files --> test this out properly!
    string line;
    while (!pt_file.eof()){
      getline(pt_file, line); 
      initPhraseFromFile(line, PL);
      numLabeled++;
    }
  }
  pt_file.close();
  cout << "Source vocabulary size: " << vocab.size() << endl; 
  cout << "Number of phrases in phrase table: " << numLabeled << endl; 
  cout << "Number of phrases with desired phrase length " << PL << ": " << all_phrases.size() << endl; 
  cout << "Maximum target phrase length: " << get<2>(max_tgtPL) << endl; 
  cout << "Phrase pair: " << get<0>(max_tgtPL) << " ||| " << get<1>(max_tgtPL) << endl;   
}

//function that takes a line from a phrase table and initialzes, as long as the phrase is 
//of the correct phrase length. 
void Phrases::initPhraseFromFile(string line, int phrase_length){
  boost::trim(line);
  vector<string> elements = multiCharSplitter(line); 
  string srcPhr = elements[0];
  vector<string> srcTokens;
  boost::split(srcTokens, srcPhr, boost::is_any_of(" "));
  if (srcTokens.size() == phrase_length){
    Phrase* phrase = NULL;
    if (phrStr2ID.find(srcPhr) == phrStr2ID.end()){ //new phrase
      for (unsigned int i = 0; i < srcTokens.size(); i++){ //add unigrams to vocab if not seen before
	if (vocab.find(srcTokens[i]) == vocab.end()){ 
	  int id = vocab.size();
	  vocab[srcTokens[i]] = id;
	}
      }
      int phrID = all_phrases.size();
      phrase = new Phrase(phrID, srcPhr, true); 
      phrStr2ID[srcPhr] = phrID;
      all_phrases.push_back(phrase);
    }
    else 
      phrase = all_phrases[phrStr2ID[srcPhr]];
    addLabel(phrase, elements); 
  }
}

//utility function to split a string according to a multi-character delimiter
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

//given a source phrase and a line from the phrase table, this function does the appropriate
//book-keeping to add the label.  
void Phrases::addLabel(Phrase* phrase, vector<string> elements){
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
  vector<string> tgtTokens;
  int maxPL;
  tie(ignore, ignore, maxPL) = max_tgtPL;  
  boost::split(tgtTokens, tgtPhr, boost::is_any_of(" "));
  if (tgtTokens.size() > maxPL)
    max_tgtPL = make_tuple(phrase->phrase_str, tgtPhr, tgtTokens.size()); 
}
  
