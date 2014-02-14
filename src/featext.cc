#include "featext.h"
#include "phrases.h"
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <set>
#include <algorithm>
#include <functional>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <math.h>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>

using namespace std;
using namespace Eigen; 
namespace fs = boost::filesystem;
namespace io = boost::iostreams;

FeatureExtractor::FeatureExtractor(){
  stop_words = set<unsigned int>(); 
  featStr2ID = map<string, unsigned int>();
  //inverted_idx = unordered_map<unsigned int, set<unsigned int> >();  
  inverted_idx = map<unsigned int, set<unsigned int> >();  
  featMat_triplets = vector<triplet>(); 
  feature_matrix = SparseMatrix<double,RowMajor>();
}

//will it use default deconstructor if I don't write?
FeatureExtractor::~FeatureExtractor(){  
}

void FeatureExtractor::writeToFile(const string featMatLoc, const string invIdxLoc){
  ofstream outFileInvIdx(invIdxLoc.c_str()); 
  boost::archive::binary_oarchive oa(outFileInvIdx); //if this works, update phrases.cc too
  oa << inverted_idx; 
  outFileInvIdx.close();  
  //if need be, we can write out featStr2ID as well
  saveMarket(feature_matrix, featMatLoc); 
}

void FeatureExtractor::readFromFile(const string featMatLoc, const string invIdxLoc){
  ifstream inFileInvIdx(invIdxLoc.c_str());
  boost::archive::binary_iarchive ia(inFileInvIdx); 
  ia >> inverted_idx; 
  inFileInvIdx.close(); 
  loadMarket(feature_matrix, featMatLoc); 
}

vector<string> FeatureExtractor::filterSentences(const string mono_dir_loc, Phrases* phrases, const unsigned int minPL, const unsigned int maxPL, const unsigned int maxPhrCount, const string monolingual_out){  
  map<string, unsigned int> unlabeled_count = map<string, unsigned int>();
  typedef map<string, unsigned int>::const_iterator const_it; 
  vector<Phrases::Phrase*> unlabeled_phrases = phrases->getUnlabeledPhrases();
  for (unsigned int i = 0; i < unlabeled_phrases.size(); i++ ) //initialize counts to 0
    unlabeled_count[unlabeled_phrases[i]->phrase_str] = 0;
  cout << "Number of unique unlabeled phrases: " << unlabeled_count.size() << endl; 
  vector<string> keys_to_search; 
  for (const_it it = unlabeled_count.begin(); it != unlabeled_count.end(); it++) //extract keys for intersection
    keys_to_search.push_back(it->first); 
  sort(keys_to_search.begin(), keys_to_search.end()); 
  fs::path dirPath(mono_dir_loc.c_str());
  if (fs::is_directory(dirPath)){
    ofstream filtered_sentences;
    filtered_sentences.open(monolingual_out.c_str()); 
    vector<string> filenames; 
    fs::directory_iterator dir_iter(dirPath), dir_end;  
    for (; dir_iter != dir_end; dir_iter++){ //convert dir_iter to vector of strings for parallel computation
      if (dir_iter->path().extension() == ".gz")
	filenames.push_back(dir_iter->path().native()); 
      else { 
	cerr << "Monolingual files need to be in .gz format" << endl; 
	filtered_sentences.close(); 
	exit(0); 
      }	
    }
    unsigned int numSentences = 0;
    omp_lock_t lock;
    omp_init_lock(&lock); //initializes the lock
    #pragma omp parallel for
    for (unsigned int i = 0; i < filenames.size(); i++){
      ifstream mono_file;
      mono_file.exceptions(ios::failbit | ios::badbit);
      mono_file.open(filenames[i].c_str(), ios_base::in | ios_base::binary);
      io::filtering_stream<io::input> decompressor; 
      decompressor.push(io::gzip_decompressor());
      decompressor.push(mono_file); 
      for (string line; getline(decompressor, line);){
	boost::trim(line); 
	vector<string> ngrams = vector<string>();
	for (unsigned int j = minPL; j < maxPL + 1; j++){ //extract ngrams of all orders
	  vector<ngram_triple> order_ngrams = extractNGrams(j, line); 
	  string ngram; 	  
	  if (order_ngrams.size() > 0){
	    for (unsigned int k = 0; k < order_ngrams.size(); k++){ //strip unnecessary info and push back
	      tie(ngram, ignore, ignore) = order_ngrams[k]; 
	      ngrams.push_back(ngram); 
	    }
	  }
	} //have all the ngrams in the line
	sort(ngrams.begin(), ngrams.end());
	vector<string> unlabeled_in_line; 
	set_intersection(keys_to_search.begin(), keys_to_search.end(), ngrams.begin(), ngrams.end(), back_inserter(unlabeled_in_line)); 
	if (unlabeled_in_line.size() > 0){ //i.e., at least one hit on this line
          #pragma omp critical(writeLineUpdateCount)
	  {
	    filtered_sentences << line << endl; 
	    numSentences++; 
	    for (unsigned int j = 0; j < unlabeled_in_line.size(); j++)
	      unlabeled_count[unlabeled_in_line[j]]++;
	  }
	}
      }
      decompressor.pop();
      mono_file.close();
      decompressor.pop();
      omp_set_lock(&lock); 
      keys_to_search.clear();      
      unsigned int numUncovered = 0; 
      for (const_it it = unlabeled_count.begin(); it != unlabeled_count.end(); it++){
	if (it->second < maxPhrCount)
	  keys_to_search.push_back(it->first);
	if (it->second == 0)
	  numUncovered++; 
      }
      sort(keys_to_search.begin(), keys_to_search.end()); //sort it here so that next time we compute intersection no need to sort
      omp_unset_lock(&lock); 
      #pragma omp critical(writeStatsToStdOut)
      {
	cout << "File " << filenames[i] << " complete; Number of sentences accumulated: " << numSentences << endl; 
	cout << "Number of uncovered phrases: " << numUncovered << " out of " << unlabeled_count.size() << endl; 
      }
    }
    omp_destroy_lock(&lock); 
    vector<string> unlabeled_hits = vector<string>();
    for (const_it it = unlabeled_count.begin(); it != unlabeled_count.end(); it++){
      if (it->second > 0)
	unlabeled_hits.push_back(it->first); 
    }
    return unlabeled_hits; 
  }
  else {
    cerr << "Path defined in 'source_mono_dir' or 'target_mono_dir' is not a directory!" << endl; 
    exit(0); 
  }
}

void FeatureExtractor::readStopWords(const string filename, const unsigned int num_sw){  
  ifstream stopwordsFile(filename.c_str()); 
  if (stopwordsFile.is_open()){
    string line;
    while (getline(stopwordsFile, line)){
      if (stop_words.size() > 2*num_sw)
	break;
      boost::trim(line); 
      vector<string> elements; 
      boost::split(elements, line, boost::is_any_of("\t")); 
      assert(elements.size() == 2);
      if (elements[0] == "<s>" || elements[0] == "</s>")
	continue;      
      stop_words.insert(getSetFeatureID(elements[0], Left)); 
      stop_words.insert(getSetFeatureID(elements[0], Right)); 
    }
    stopwordsFile.close(); 
  }
}

void FeatureExtractor::extractFeatures(Phrases* phrases, const string mono_filename, const unsigned int winsize, const unsigned int minPL, const unsigned int maxPL){
  ifstream monoFile(mono_filename.c_str()); 
  if (monoFile.is_open()){
    string line;
    while (getline(monoFile, line)){
      boost::trim(line); 
      vector<string> sentence;
      boost::split(sentence, line, boost::is_any_of(" ")); 
      for (unsigned int ngram_order = minPL; ngram_order < maxPL + 1; ngram_order++){
	vector<ngram_triple> order_ngrams = extractNGrams(ngram_order, line); 	
	string ngram;
	unsigned int left_idx;
	unsigned int right_idx; 
	for (unsigned int i = 0; i < order_ngrams.size(); i++){
	  tie(ngram, left_idx, right_idx) = order_ngrams[i]; 
	  int phrID = phrases->getPhraseID(ngram); 
	  if (phrID > -1){ //i.e., phrase exists in our list of phrases
	    phrases->getNthPhrase(phrID)->count++; //is this a good idea? to modify directly? 
	    int startIdx = 0; 
	    int endIdx = left_idx; 
	    if (left_idx > winsize)
	      startIdx = left_idx-winsize; 
	    if (left_idx > 0){
	      vector<string> subsent(sentence.begin()+startIdx, sentence.begin()+endIdx); 	      
	      addContext(phrID, subsent, Left); 
	    }
	    startIdx = right_idx+1;
	    endIdx = sentence.size()-1;
	    if (sentence.size()-1-right_idx > winsize)
	      endIdx = winsize+startIdx; 
	    if (sentence.size()-1-right_idx > 0){
	      vector<string> subsent(sentence.begin()+startIdx, sentence.begin()+endIdx); 
	      addContext(phrID, subsent, Right); 
	    }
	  }	  
	}	
      }
    }
    monoFile.close(); 
  }
  const unsigned int numTotalPhrases = phrases->getNumUnlabeledPhrases() + phrases->getNumLabeledPhrases();
  feature_matrix.resize(numTotalPhrases, featStr2ID.size()); 
  feature_matrix.reserve(featMat_triplets.size()); 
  feature_matrix.setFromTriplets(featMat_triplets.begin(), featMat_triplets.end()); 
  featMat_triplets.clear(); 
  cout << "Co-occurrence counts assembled into feature matrix, with dimensions " << numTotalPhrases << " x " << featStr2ID.size() << endl; 
}

void FeatureExtractor::addContext(const unsigned int phraseID, vector<string> subsent, const ContextSide side){
  vector<unsigned int> contextFeatureIDs; 
  for (unsigned int i = 0; i < subsent.size(); i++)
    contextFeatureIDs.push_back(getSetFeatureID(subsent[i], side)); 
  assert(contextFeatureIDs.size() == subsent.size()); 
  for (unsigned int i = 0; i < contextFeatureIDs.size(); i++){
    unsigned int contextID = contextFeatureIDs[i]; 
    if (stop_words.find(contextID) == stop_words.end()){ //only add to inverted index if feature is not stop word
      invIdxIter it = inverted_idx.find(contextID); 
      if (it != inverted_idx.end())
	it->second.insert(phraseID); 
      else {
	set<unsigned int> phraseIDs = set<unsigned int>();
	phraseIDs.insert(phraseID);
	inverted_idx[contextID] = phraseIDs; 
      } 
      //featMat_triplets.push_back(triplet(phraseID, contextID, 1.0)); //over here, we only add the feature if it is not a stop word
    }    
    featMat_triplets.push_back(triplet(phraseID, contextID, 1.0)); 
  }
}

void FeatureExtractor::rescaleCoocToPMI(){
  cout << "Converting co-occurrence counts to PMI values" << endl;   
  //VectorXd indFeatSumRowInv = (feature_matrix*VectorXd::Ones(feature_matrix.cols())).array().inverse(); //compute sum over features for each phrase, and take reciprocal
  VectorXd indFeatSumRowInv = (feature_matrix*VectorXd::Ones(feature_matrix.cols())).cwiseInverse(); 
  SparseMatrix<double,RowMajor> left_mult(feature_matrix.rows(), feature_matrix.rows()); 
  vector<triplet> left_mult_diagonal = vector<triplet>();
  for (unsigned int i = 0; i < indFeatSumRowInv.size(); i++) //place inverse sum on diagonal
    left_mult_diagonal.push_back(triplet(i, i, indFeatSumRowInv[i])); 
  left_mult.reserve(left_mult_diagonal.size()); 
  left_mult.setFromTriplets(left_mult_diagonal.begin(), left_mult_diagonal.end());   
  SparseMatrix<double,RowMajor> PMI(feature_matrix.rows(), feature_matrix.cols()); 
  PMI.reserve(feature_matrix.nonZeros()); 
  PMI = left_mult * feature_matrix; 
  //for memory efficiency, may want to clear left_mult, left_mult_diagonal here
  cout << "Computed Prob(feature | phrase)" << endl; 
  double allFeatureSum = 0.0; 
  for (int i = 0; i < feature_matrix.outerSize(); i++){ //sum over all feature values
    for (SparseMatrix<double,RowMajor>::InnerIterator it(feature_matrix,i); it; ++it){
      allFeatureSum += it.value(); 
    }
  }
  RowVectorXd indFeatSumColInv = ((RowVectorXd::Ones(feature_matrix.rows())*feature_matrix).array()*(1.0/allFeatureSum)).cwiseInverse();
  SparseMatrix<double> right_mult(feature_matrix.cols(), feature_matrix.cols()); 
  vector<triplet> right_mult_diagonal = vector<triplet>();
  for (int i = 0; i < indFeatSumColInv.size(); i++)
    right_mult_diagonal.push_back(triplet(i,i,indFeatSumColInv[i]));
  right_mult.reserve(right_mult_diagonal.size());
  right_mult.setFromTriplets(right_mult_diagonal.begin(), right_mult_diagonal.end());
  PMI = PMI * right_mult; 
  cout << "Computed Prob(feature | phrase) / P(feature)" << endl; 
  for (int i = 0; i < PMI.outerSize(); i++){ //convert result to log ratio
    for (SparseMatrix<double,RowMajor>::InnerIterator it(PMI,i); it; ++it)
      PMI.coeffRef(it.row(), it.col()) = log(it.value()); 
  }
  feature_matrix = PMI; //assume deep copy? 
  cout << "Finished conversion to PMI" << endl; 
  //PMI, right mult, left_mult are local and will be cleaned up here  
}

//doesn't seem to be fully working? try compress maybe? 
void FeatureExtractor::pruneFeaturesByCount(const unsigned int minCount){
  int nnzs = feature_matrix.nonZeros();
  cout << "Initially: " << nnzs << " non-zero elements in feature matrix" << endl; 
  for (int i = 0; i < feature_matrix.outerSize(); i++){
    for (SparseMatrix<double,RowMajor>::InnerIterator it(feature_matrix,i); it; ++it){
      if (it.value() < minCount)
	feature_matrix.coeffRef(it.row(), it.col()) = 0;
    }      
  }
  feature_matrix.prune([](int i, int j, double val){ return val > 0; }); 
  int nnzs_after_prune = feature_matrix.nonZeros();
  cout << "After pruning features with count less than " << minCount << ", there are " << nnzs_after_prune << " non-zero elements in feature matrix" << endl; 
}

void FeatureExtractor::analyzeFeatureMatrix(const vector<Phrases::Phrase*> unlabeled_phrases){
  set<int> unlabeled_ids = set<int>();
  for (unsigned int i = 0; i < unlabeled_phrases.size(); i++)
    unlabeled_ids.insert(unlabeled_phrases[i]->id);
  int zr_lab = 0, zr_unl = 0, zr_fil_lab = 0, zr_fil_unl = 0; 
  for (int i = 0; i < feature_matrix.rows(); i++){
    if (feature_matrix.row(i).nonZeros() == 0){ //if this doesn't work, then use innerVector construction or can also access InnerNNZs? 
      if (unlabeled_ids.find(i) == unlabeled_ids.end()) //then its a labeled phrase
	zr_lab++;
      else
	zr_unl++;
    }
    vector<int> filtered_ids = vector<int>(); 
    for (SparseMatrix<double,RowMajor>::InnerIterator it(feature_matrix,i); it; ++it){
      if (stop_words.find(it.col()) == stop_words.end()) //not in stop words list
	filtered_ids.push_back(it.col()); 
    }
    if (filtered_ids.size() == 0){
      if (unlabeled_ids.find(i) == unlabeled_ids.end())
	zr_fil_lab++;
      else
	zr_fil_unl++; 
    }
  }
  cout << "Number of phrases (labeled+unlabeled) in feature matrix: " << feature_matrix.rows() << " (" << feature_matrix.rows() - unlabeled_phrases.size() << " + " << unlabeled_phrases.size() << ")" << endl; 
  cout << "Number of featureless labeled phrases: " << zr_lab << endl; 
  cout << "Number of featureless labeled phrases after stop-word filtering: " << zr_fil_lab << endl; 
  cout << "Number of featureless unlabeled phrases: " << zr_unl << endl; 
  cout << "Number of featureless unlabeled phrases after stop-word filtering: " << zr_fil_unl << endl; 
}

unsigned int FeatureExtractor::getSetFeatureID(string featStr, const ContextSide side){
  featStr += (side == Left) ? "_L" : "_R"; 
  if (featStr2ID.find(featStr) == featStr2ID.end()){
    unsigned int id = featStr2ID.size();
    featStr2ID[featStr] = id;
    return id; 
  }
  else
    return featStr2ID[featStr];
}

vector<ngram_triple> FeatureExtractor::extractNGrams(const unsigned int n, const string str){
  vector<ngram_triple> ngrams = vector<ngram_triple>();
  vector<string> words; 
  boost::split(words, str, boost::is_any_of(" ")); //tokenizes the input sentence
  if (n > words.size())
    return ngrams; 
  else {
    for (unsigned int i = 0; i < words.size() - n + 1; i++ ){
      ngram_triple triple = make_tuple(concat(words, i, i+n), i, i + n - 1); 
      ngrams.push_back(triple); 
    }
    return ngrams;
  }
}

string FeatureExtractor::concat(vector<string> words, const unsigned int start, const unsigned int end){
  string ngram = "";
  for (unsigned int i = start; i < end; i++)
    ngram += ((i > start) ? " " : "") + words[i]; 
  return ngram; 
}
