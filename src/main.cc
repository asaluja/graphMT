#include <iostream>
#include <vector>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>
#include <time.h>
#include "options.h"
#include "phrases.h"

using namespace std;
namespace po = boost::program_options;

int main(int argc, char** argv){
  cout << "Graph Propagation for Phrase Table Expansion" << endl; 
  cout << "Avneesh Saluja (avneesh@cs.cmu.edu), 2014" << endl; 
  Options* opts = new Options(argc, argv); //reads in config file
  po::variables_map conf = opts->getConf();
  Phrases* src_phrases = new Phrases();
  cout << "Reading in phrase table" << endl; 
  clock_t start = clock();
  src_phrases->addLabeledPhrasesFromFile(conf["phrase_table"].as<string>(), conf["phrase_length"].as<int>());
  clock_t end = clock();
  double timeTaken = ((double)(end-start)) / ((double) CLOCKS_PER_SEC); 
  cout << "Time taken: " << timeTaken << " seconds" << endl; 
  src_phrases->normalizeLabelDistributions();
  src_phrases->addUnlabeledPhrasesFromFile(conf["evaluation_corpus"].as<string>(), conf["phrase_length"].as<int>(), conf["write_unlabeled"].as<string>(), conf.count("analyze_unlabeled")); 
  Phrases* tgt_phrases = new Phrases(src_phrases); 
  return 0;
}
