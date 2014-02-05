#include "featext.h"
#include "phrases.h"
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

using namespace std;
namespace fs = boost::filesystem;
namespace io = boost::iostreams;

//will it use default constructor if I don't write? 
FeatureExtractor::FeatureExtractor(){
}

FeatureExtractor::~FeatureExtractor(){  
}

vector<string> FeatureExtractor::filterSentences(const string mono_dir_loc, Phrases* phrases, const unsigned int minPL, const unsigned int maxPL, const unsigned int maxPhrCount, const string monolingual_out){  
  map<string, unsigned int> unlabeled_count = map<string, unsigned int>();
  typedef map<string, unsigned int>::const_iterator const_it; 
  vector<Phrases::Phrase*> unlabeled_phrases = phrases->getUnlabeledPhrases();
  for (unsigned int i = 0; i < unlabeled_phrases.size(); i++ ) //initialize counts to 0
    unlabeled_count[unlabeled_phrases[i]->phrase_str] = 0;
  cout << "Number of unlabeled phrases: " << unlabeled_phrases.size() << endl; 
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
