#include "phrases.h"
#include "featext.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include <set>
#include <math.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/lexical_cast.hpp>

namespace io = boost::iostreams;
namespace fs = boost::filesystem;
using namespace std;
const string delimiter = " ||| "; 

//standard constructor
Phrases::Phrases(){
  all_phrases = vector<Phrase*>();
  phrStr2ID = map<string, unsigned int>();
  label_phrStr2ID = map<string, unsigned int>();
  label_phrID2Str = map<unsigned int, string>(); 
  vocab = map<string, unsigned int>();
  max_tgtPL = make_tuple("", "", 0); 
  numLabeled = 0, numUnlabeled = 0; 
}

//constructor for initializing phrases of other side
Phrases::Phrases(const Phrases* orig_phrases){
  all_phrases = vector<Phrase*>();
  all_phrases.resize(orig_phrases->label_phrStr2ID.size()); 
  phrStr2ID = map<string, unsigned int>(orig_phrases->label_phrStr2ID);
  label_phrStr2ID = map<string, unsigned int>(); //dummy DS for target phrases
  label_phrID2Str = map<unsigned int, string>(); //dummy DS for target phrases
  vocab = map<string, unsigned int>(); //dummy for target phrases
  max_tgtPL = make_tuple("", "", 0);
  numLabeled = 0, numUnlabeled = 0; 
  typedef map<string,unsigned int>::const_iterator iter;
  for (iter it = phrStr2ID.begin(); it != phrStr2ID.end(); it++){ //adding target phrases as Phrase structs
    vector<string> tokens; 
    boost::split(tokens, it->first, boost::is_any_of(" "));    
    Phrase* phrase = new Phrase(it->second, it->first, false); 
    all_phrases[it->second] = phrase; 
    numUnlabeled++; 
  }
  cout << "Number of phrases: " << numLabeled + numUnlabeled << endl; 
}

Phrases::~Phrases(){
  for (unsigned int i = 0; i < all_phrases.size(); i++ )
    delete all_phrases[i];
}

void Phrases::writePhraseIDsToFile(const string filename, const bool writeLabeled){
  if (writeLabeled)
    cerr << "Error: 'writeLabeled' = true for writePhraseIDs not implemented" << endl; 
  else {
    ofstream phraseIDs;
    phraseIDs.open(filename.c_str()); 
    assert(phraseIDs != NULL); 
    vector<Phrase*> unlabeled_phrases = getUnlabeledPhrases(); 
    for (unsigned int i = 0; i < unlabeled_phrases.size(); i++)
      phraseIDs << unlabeled_phrases[i]->phrase_str << delimiter << unlabeled_phrases[i]->id << endl; 
    phraseIDs.close();
  }
}

void Phrases::readPhraseIDsFromFile(const string filename, const bool readLabeled){
  if (readLabeled)
    cerr << "Error: 'readLabeled' = true for readPhraseIDs not implemented" << endl; 
  else { 
    ifstream phraseIDs(filename.c_str());
    if (phraseIDs.is_open()){
      string line;
      while (getline(phraseIDs, line)){
	boost::trim(line);
	vector<string> elements = multiCharSplitter(line);
	assert(elements.size() == 2);
	vector<string> tokens;
	boost::split(tokens, elements[0], boost::is_any_of(" ")); 
	if (phrStr2ID.find(elements[0]) == phrStr2ID.end()) //i.e., we have not taken the label from the phrase table, it is a generated label that we are reading from file
	  initPhrase(elements[0], tokens, atoi(elements[1].c_str()), false); 
      }
      phraseIDs.close();    
    }
    else { cerr << "Could not read phrase IDs at location " << filename << endl; exit(0); }
  }
  cout << "Total number of phrases now: " << numLabeled + numUnlabeled << endl; 
}

//N.B.: temporary change to this function - revert back later
void Phrases::readLabelPhraseIDsFromFile(const string filename){
  ifstream phraseIDs(filename.c_str()); 
  if (phraseIDs.is_open()){
    string line; 
    while (getline(phraseIDs, line)){
      boost::trim(line);
      vector<string> elements = multiCharSplitter(line); 
      assert(elements.size() == 2); 
      if (label_phrStr2ID.find(elements[0]) == label_phrStr2ID.end()){ //new candidate in label space
	assert(atoi(elements[1].c_str()) > 0); 
	label_phrStr2ID[elements[0]] = atoi(elements[1].c_str()); //hopefully this is the same as above? how to check
	label_phrID2Str[atoi(elements[1].c_str())] = elements[0]; 
      }
    }
    phraseIDs.close(); 
    cout << "Total number of label phrases now: " << label_phrStr2ID.size() << endl; 
  }
  else { cerr << "Could not read phrase IDs for labels at location " << filename << endl; exit(0); }
}

void Phrases::writePhraseTable(Phrases* tgt_phrases, const string pt_format, const string new_pt_loc, LexicalScorer* const lex){
  ofstream out(new_pt_loc.c_str());  
  assert(out != NULL); 
  vector<Phrase*> unlabeled_phrases = getUnlabeledPhrases();
  int num_src_marginal_pos = 0;
  int num_tgt_marginal_pos = 0; 
  int num_prob_pos = 0; 
  cout << "Number of unlabeled phrases to write out: " << unlabeled_phrases.size() << endl; 
  for (unsigned int i = 0; i < unlabeled_phrases.size(); i++){
    Phrase* phrase = unlabeled_phrases[i]; 
    if (phrase->marginal > 0){ //otherwise we have not seen the phrase in the monolingual corpus at all
      num_src_marginal_pos++; 
      set<int> labels = phrase->getLabels(); 
      vector<string> srcPhrases = vector<string>();
      vector<string> tgtPhrases = vector<string>(); 
      vector<pair<double, double> > fwd_bwd_prob = vector<pair<double, double> >(); 
      for (set<int>::iterator it = labels.begin(); it != labels.end(); it++){
	if (tgt_phrases->getNthPhrase(*it)->marginal > 0){
	  num_tgt_marginal_pos++; 
	  srcPhrases.push_back(phrase->phrase_str); 
	  tgtPhrases.push_back(getLabelPhraseStr(*it)); 
	  double fwd_prob = phrase->label_distribution[*it];
	  if (fwd_prob == 0)
	    cout << "Phrase pair '" << phrase->phrase_str << " ||| " << getLabelPhraseStr(*it) << "' with IDs (" << phrase->id << "," << *it <<") has P(e|f) = 0" << endl; 
	  double bwd_prob = fwd_prob * (phrase->marginal / tgt_phrases->getNthPhrase(*it)->marginal); 
	  fwd_bwd_prob.push_back(make_pair(fwd_prob, bwd_prob)); 
	}
      }
      vector<pair<double, double> > lex_scores = lex->scorePhrasePairs(srcPhrases, tgtPhrases); 
      //need to incorporate cdec style grammar writing here, but for now just work with moses
      assert(pt_format == "moses"); //moses order is P(f|e) lex(f|e) P(e|f) lex(e|f)
      for (unsigned int j = 0; j < srcPhrases.size(); j++){
	if (fwd_bwd_prob[j].first > 0){
	  num_prob_pos++; 
	  string output_line = srcPhrases[j] + " ||| " + tgtPhrases[j] + " ||| " + boost::lexical_cast<string>(fwd_bwd_prob[j].second) + " " + boost::lexical_cast<string>(lex_scores[j].second) + " " + boost::lexical_cast<string>(fwd_bwd_prob[j].first) + " " + boost::lexical_cast<string>(lex_scores[j].first) + " ||| ";
	  out << output_line << endl; 
	}
	//don't think i need to do anything with alignments or counts      
      }
    }
  }
  out.close(); 
  cout << "Number of source marginal positive phrases: " << num_src_marginal_pos << endl; 
  cout << "Number of target marginal positive phrases out of valid phrase pairs: " << num_tgt_marginal_pos << endl; 
  cout << "Number of valid lexical score phrase pairs: " << num_prob_pos << endl; 
}

//goes through label distribution for each labeled soure phrase and normalizes (sum = 1)
void Phrases::normalizeLabelDistributions(){  
  for (unsigned int i = 0; i < all_phrases.size(); i++ )
    all_phrases[i]->normalizeDistribution(); 
}

//function primarily meant for debugging
void Phrases::printLabels(const string phrase){
  int phrID = phrStr2ID[phrase];
  map<int,double> labels = all_phrases[phrID]->label_distribution; 
  typedef map<int,double>::const_iterator it;
  for (it i = labels.begin(); i != labels.end(); i++)
    cout << "Key: " << i->first << "; Value: " << i->second << endl; 
}

void Phrases::addGeneratedPhrases(const vector<string> generated_phrases){
  for (unsigned int i = 0; i < generated_phrases.size(); i++){
    vector<string> tokens; 
    boost::split(tokens, generated_phrases[i], boost::is_any_of(" "));    
    if (phrStr2ID.find(generated_phrases[i]) == phrStr2ID.end()) //i.e., we have not taken the label from the phrase table, it is a generated label that we are reading
      initPhrase(generated_phrases[i], tokens, all_phrases.size(), false); 
  }
}

//this format assumes a cdec/moses decoder style output format
//for the mbest-list phrases (delimited by ' ||| ')
int Phrases::readMBestListFromFile(const string filename_in, const string filename_out, const vector<Phrase*> unlabeled_phrases){
  unsigned int maxPL = 0; 
  map<const string, vector<string> > mbest_by_src = map<const string, vector<string> >();
  typedef map<const string, vector<string> >::iterator iter;
  ifstream mbest_list(filename_in.c_str()); 
  string line;    
  if (mbest_list.is_open()){
    while (getline(mbest_list, line)){    
      boost::trim(line); 
      vector<string> elements = multiCharSplitter(line); 
      assert(elements.size() > 2); 
      const string srcPhr = unlabeled_phrases[atoi(elements[0].c_str())]->phrase_str; 
      string mbest_hyp = elements[1];
      boost::trim(mbest_hyp); 
      vector<string> tgtTokens;
      boost::split(tgtTokens, mbest_hyp, boost::is_any_of(" "));    
      if (tgtTokens.size() > maxPL)
	maxPL = tgtTokens.size();
      Phrase* phrase = new Phrase(all_phrases.size(), mbest_hyp, false);     
      all_phrases.push_back(phrase); 
      iter mbest_vec = mbest_by_src.find(srcPhr); 
      if (mbest_vec == mbest_by_src.end()){
	vector<string> mbest_phrases_for_src = vector<string>();
	mbest_phrases_for_src.push_back(mbest_hyp); 
	mbest_by_src[srcPhr] = mbest_phrases_for_src; 
      }
      else
	mbest_vec->second.push_back(mbest_hyp); 
      numUnlabeled++; 
    }
    mbest_list.close(); 
  }
  else { cerr << "Could not open mbest list at location: " << filename_in << endl; exit(0); }
  cout << "Total number of mbest list candidates generated: " << all_phrases.size() << endl; 
  ofstream outFile(filename_out.c_str());
  assert (outFile != NULL); 
  boost::archive::text_oarchive oa(outFile); 
  oa << mbest_by_src; 
  outFile.close(); 
  return maxPL; 
}

map<const string, vector<string> > Phrases::readFormattedMBestListFromFile(const string mbest_processed_loc){
  ifstream inFileMBestMap(mbest_processed_loc.c_str()); 
  if (inFileMBestMap.good()){
  map<const string, vector<string> > mbest_by_src = map<const string, vector<string> >();
  boost::archive::text_iarchive ia(inFileMBestMap); 
  ia >> mbest_by_src; 
  inFileMBestMap.close();
  return mbest_by_src;   
  }
  else { cerr << "Could not read formatted m-best list from location " << mbest_processed_loc << endl; exit(0); }
}

void::Phrases::computeMarginals(const string cooc_loc){
  SparseMatrix<double,RowMajor> cooc_matrix = SparseMatrix<double,RowMajor>();   
  loadMarket(cooc_matrix, cooc_loc); 
  VectorXd indFeatSumRow = cooc_matrix*VectorXd::Ones(cooc_matrix.cols()); //sum over features for each phrase
  assert(indFeatSumRow.size() == all_phrases.size());   
  double normalizer = indFeatSumRow.sum(); 
  for (unsigned int i = 0; i < indFeatSumRow.size(); i++){
    all_phrases[i]->marginal = indFeatSumRow[i] / normalizer; 
  }
}

//goes through evaluation corpus, first extracts all n-grams of length PL, and then adds
//n-grams that aren't in phrase tablea s unlabeled n-grams. 
void Phrases::addUnlabeledPhrasesFromFile(const string filename, const unsigned int PL, const string out_filename, const bool analyze){
  map<const string, unsigned int> ngram_count = map<const string, unsigned int>();  
  ifstream eval_corpus(filename.c_str());
  if (eval_corpus.good()){
    string line;
    if (eval_corpus.is_open()){
      while (getline(eval_corpus, line)){
	boost::trim(line);
	const vector<ngram_triple> ngrams_from_line = FeatureExtractor::extractNGrams(PL, line); 
	string ngram;
	for (unsigned int i = 0; i < ngrams_from_line.size(); i++ ){
	  tie(ngram, ignore, ignore) = ngrams_from_line[i]; 
	  pair<map<const string, unsigned int>::iterator,bool> ret; 
	  ret = ngram_count.insert(pair<const string, unsigned int>(ngram, 1)); 
	  if (ret.second == false)
	    ngram_count[ngram]++;
	}
      }
      eval_corpus.close();
    }
    cout << "Number of " << PL << "-grams in evaluation corpus: " << ngram_count.size() << endl;   
  }
  else { cerr << "Could not find evaluation corpus at " << filename << endl; exit(0); }
  ofstream unlabeled_phrases;
  unlabeled_phrases.open(out_filename.c_str()); //write out unlabeled n-grams
  if (unlabeled_phrases != NULL){
    typedef map<const string, unsigned int>::const_iterator iter;
    for (iter it = ngram_count.begin(); it != ngram_count.end(); it++ ){
      const string srcPhr = it->first;
      iter checkUnlabeled = phrStr2ID.find(srcPhr);
      if (checkUnlabeled == phrStr2ID.end()){ //add phrases not in phrase table
	vector<string> srcTokens;
	boost::split(srcTokens, srcPhr, boost::is_any_of(" ")); 
	initPhrase(srcPhr, srcTokens, all_phrases.size(), false); 
	unlabeled_phrases << srcPhr << endl; 
      }
    }
    cout << "Number of unlabeled " << PL << "-grams in evaluation corpus: " << numUnlabeled << endl; 
    unlabeled_phrases.close();
  }
  else { cerr << "Could not write unlabeled phrases to location " << out_filename << endl; exit(0); }
  if (analyze) //if defined, we can analyze the phrases
    analyzeUnlabeledPhrases(ngram_count);
}

//for analysis purposes: breaks down unknown n-grams into all known unigrams, some known, or none known.  
void Phrases::analyzeUnlabeledPhrases(map<const string, unsigned int>& ngram_count){
  vector<Phrase*> unlabeled_phrases = getUnlabeledPhrases(); 
  int kk = 0, kkt = 0, uk = 0, ukt = 0, uu = 0, uut = 0;
  typedef map<string, unsigned int>::const_iterator iter;
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
      set<bool>::const_iterator val = is_word_oov.begin();
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
void Phrases::addLabeledPhrasesFromFile(const string filename, const unsigned int PL, const string format){
  ifstream pt_file(filename.c_str()); //file handle for phrase table
  unsigned int numPhrases = 0; 

  if (pt_file.good()){
    const fs::path p(filename); //do we need to clean this up somewhere? 
    if (p.extension() == ".gz"){ //special handling for .gz files
      io::filtering_stream<io::input> decompressor;
      decompressor.push(io::gzip_decompressor());
      decompressor.push(pt_file);
      for (string line; getline(decompressor, line);){
	initPhraseFromFile(line, PL, format); 
	numPhrases++; 
      }
      pt_file.close(); 
    }
    else { //for non .gz files --> test this out properly!
      string line;
      if (pt_file.is_open()){
	while (getline(pt_file, line)){
	  initPhraseFromFile(line, PL, format);
	  numPhrases++; 
	}
	pt_file.close(); 
      }
    }
  }
  else { 
    cerr << "Cannot find phrase table at " << filename << endl; 
    exit(0); 
  }
  cout << "Source vocabulary size: " << vocab.size() << endl; 
  cout << "Number of phrases in phrase table: " << numPhrases << endl; 
  cout << "Number of phrases with desired phrase length " << PL << ": " << all_phrases.size() << endl; 
  cout << "Maximum target phrase length: " << get<2>(max_tgtPL) << endl; 
  cout << "Phrase pair: " << get<0>(max_tgtPL) << " ||| " << get<1>(max_tgtPL) << endl;   
}

//function that takes a line from a phrase table and initialzes, as long as the phrase is 
//of the correct phrase length. 
void Phrases::initPhraseFromFile(string line, const unsigned int phrase_length, const string format){
  boost::trim(line);
  vector<string> elements = multiCharSplitter(line); 
  const string srcPhr = (format == "cdec") ? elements[1] : elements[0];   
  vector<string> srcTokens; //initialize this maybe? 
  boost::split(srcTokens, srcPhr, boost::is_any_of(" "));
  if (srcTokens.size() == phrase_length){
    Phrase* phrase = (phrStr2ID.find(srcPhr) == phrStr2ID.end()) ? initPhrase(srcPhr, srcTokens, all_phrases.size(), true) : all_phrases[phrStr2ID[srcPhr]]; 
    if (format == "cdec")
      addLabelCdec(phrase, elements);
    else
      addLabelMoses(phrase, elements); 
  }
}
 
Phrases::Phrase* Phrases::initPhrase(const string phr, const vector<string> tokens, const int phrID, bool isLabeled){
  if (isLabeled){ //vocab is only used for unlabeled phrases analysis 
    for (unsigned int i = 0; i < tokens.size(); i++){ //add unigrams to vocab if not seen before
      if (vocab.find(tokens[i]) == vocab.end()){
	const int id = vocab.size();
	vocab[tokens[i]] = id;
      }
    }
  }
  Phrase* phrase = new Phrase(phrID, phr, isLabeled); 
  phrStr2ID[phr] = phrID;
  all_phrases.push_back(phrase); //BUG!!!!!
  if (isLabeled)
    numLabeled++;
  else
    numUnlabeled++;
  return phrase; 
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
//Note: this function is consistent with the cdec formatting
void Phrases::addLabelCdec(Phrase* phrase, vector<string> elements){
  const string tgtPhr = elements[2];
  const string featStr = elements[3];
  vector<string> featStrVec;
  double fwdPhrProb = 0; 
  boost::split(featStrVec, featStr, boost::is_any_of(" ")); 
  for (unsigned int i = 0; i < featStrVec.size(); i++ ){
    vector<string> keyValPair; 
    boost::split(keyValPair, featStrVec[i], boost::is_any_of(" ")); 
    if (keyValPair[0] == "EgivenFCoherent"){
      const double fwdPhrLogProb = -atof(keyValPair[1].c_str()); 
      fwdPhrProb = pow(10, fwdPhrLogProb); 
      break; 
    }
  }
  int label_id; 
  if (label_phrStr2ID.find(tgtPhr) == label_phrStr2ID.end()){ //new label phrase
    label_id = label_phrStr2ID.size();
    label_phrStr2ID[tgtPhr] = label_id;
    label_phrID2Str[label_id] = tgtPhr; 
  }
  else
    label_id = label_phrStr2ID[tgtPhr];
  phrase->label_distribution[label_id] = fwdPhrProb;    
  vector<string> tgtTokens;
  unsigned int maxPL;
  tie(ignore, ignore, maxPL) = max_tgtPL;  
  boost::split(tgtTokens, tgtPhr, boost::is_any_of(" "));
  if (tgtTokens.size() > maxPL)
    max_tgtPL = make_tuple(phrase->phrase_str, tgtPhr, tgtTokens.size()); 
} 

//given a source phrase and a line from the phrase table, this function does the appropriate
//book-keeping to add the label.  
//NOTE: this function is specific to the older formatting of the phrase table
void Phrases::addLabelMoses(Phrase* phrase, vector<string> elements){
  const string tgtPhr = elements[1];
  const string featStr = elements[2];
  vector<string> featStrVec;
  boost::split(featStrVec, featStr, boost::is_any_of(" "));
  //assert(featStrVec.size() == 4); //new moses phrase table format has 4 features
  const double fwdPhrProb = atof(featStrVec[2].c_str()); //in moses, features are just string, P(e|f) is third one
  int label_id; 
  if (label_phrStr2ID.find(tgtPhr) == label_phrStr2ID.end()){ //new label phrase
    label_id = label_phrStr2ID.size();
    label_phrStr2ID[tgtPhr] = label_id;
    label_phrID2Str[label_id] = tgtPhr; 
  }
  else
    label_id = label_phrStr2ID[tgtPhr];
  phrase->label_distribution[label_id] = fwdPhrProb;    
  vector<string> tgtTokens;
  unsigned int maxPL;
  tie(ignore, ignore, maxPL) = max_tgtPL;  
  boost::split(tgtTokens, tgtPhr, boost::is_any_of(" "));
  if (tgtTokens.size() > maxPL)
    max_tgtPL = make_tuple(phrase->phrase_str, tgtPhr, tgtTokens.size()); 
} 
