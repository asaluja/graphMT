#include <iostream>
#include <vector>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>
#include <time.h>
#include <omp.h>
#include "options.h"
#include "phrases.h"
#include "featext.h"

using namespace std;
namespace po = boost::program_options;
inline double duration(clock_t start, clock_t end) { return ((double)(end-start)) / ((double) CLOCKS_PER_SEC); }

int main(int argc, char** argv){  
  cout << "Graph Propagation for Phrase Table Expansion" << endl; 
  cout << "Avneesh Saluja (avneesh@cs.cmu.edu), 2014" << endl; 
  Options* opts = new Options(argc, argv); //reads in config file  
  po::variables_map conf = opts->getConf();
  unsigned int numThreads = conf["number_threads"].as<int>();
  omp_set_num_threads(numThreads); 
  Phrases* src_phrases = new Phrases();
  int pl = conf["phrase_length"].as<int>();
  cout << "Reading in phrase table" << endl; 
  clock_t start = clock();  
  src_phrases->addLabeledPhrasesFromFile(conf["phrase_table"].as<string>(), pl, conf["phrase_table_format"].as<string>());
  cout << "Time taken: " << duration(start, clock()) << " seconds" << endl; 
  src_phrases->normalizeLabelDistributions();
  src_phrases->addUnlabeledPhrasesFromFile(conf["evaluation_corpus"].as<string>(), pl, conf["write_unlabeled"].as<string>(), conf.count("analyze_unlabeled"));   
  Phrases* tgt_phrases = new Phrases(src_phrases); 
  string stage = conf["stage"].as<string>();
  transform(stage.begin(), stage.end(), stage.begin(), ::tolower);
  if (stage == "selectcorpora"){
    string side = conf["corpora_selection_side"].as<string>();
    transform(side.begin(), side.end(), side.begin(), ::tolower);
    FeatureExtractor* corpus_selector = new FeatureExtractor();
    if (side == "source"){
      cout << "Beginning corpus filtering for source side" << endl; 
      start = clock();
      corpus_selector->filterSentences(conf["source_mono_dir"].as<string>(), src_phrases, pl, pl, conf["max_phrase_count"].as<int>(), conf["source_monolingual"].as<string>());
      cout << "Time taken: " << duration(start, clock()) / numThreads << " seconds" << endl;             
    }
    else if (side == "target"){
      cout << "Beginning corpus filtering for target side" << endl; 
      start = clock();
      Phrases* mbest_phrases = new Phrases();
      int maxPL = mbest_phrases->readMBestListFromFile(conf["mbest_fromdecoder_location"].as<string>(), conf["mbest_processed_location"].as<string>(), src_phrases->getUnlabeledPhrases()); 
      if (maxPL > conf["max_target_phrase_length"].as<int>()){
	cout << "Maximum target phrase length from m-best phrases: " << maxPL << endl; 
	maxPL = conf["max_target_phrase_length"].as<int>();
	cout << "Setting to value in config file: " << conf["max_target_phrase_length"].as<int>() << endl; 
      }
      vector<string> generated_candidates = corpus_selector->filterSentences(conf["target_mono_dir"].as<string>(), mbest_phrases, 1, maxPL, conf["max_phrase_count"].as<int>(), conf["target_monolingual"].as<string>());
      cout << "Time taken: " << duration(start, clock()) / numThreads << endl; 
      cout << "Number of m-best phrases with count > 0: " << generated_candidates.size() << endl; 
      tgt_phrases->addGeneratedPhrases(generated_candidates); 
      tgt_phrases->writePhraseIDsToFile(conf["target_phraseIDs"].as<string>(), false);    
      delete mbest_phrases; 
    }
    else {
      cerr << "Incorrect argument for 'corpora_selection_side' field" << endl; 
      exit(0);
    }
    delete corpus_selector;
  }
  else if (stage == "extractfeatures"){
    cout << "Beginning source-side feature extraction" << endl; 
    start = clock();
    FeatureExtractor* source_extractor = new FeatureExtractor();
    source_extractor->readStopWords(conf["source_stopwords"].as<string>(), conf["stop_list_size"].as<int>()); 
    source_extractor->extractFeatures(src_phrases, conf["source_monolingual"].as<string>(), conf["window_size"].as<int>(), pl, pl);
    if (conf["minimum_feature_count"].as<int>() > 1)
      source_extractor->pruneFeaturesByCount(conf["minimum_feature_count"].as<int>());
    if (conf.count("analyze_feature_matrix"))
      source_extractor->analyzeFeatureMatrix(src_phrases->getUnlabeledPhrases());
    source_extractor->rescaleCoocToPMI();
    source_extractor->writeToFile(conf["source_feature_matrix"].as<string>(), conf["source_feature_extractor"].as<string>()); 
    cout << "Time taken: " << duration(start, clock()) << endl;     
    delete source_extractor; 
    /* cout << "Beginning target-side feature extraction" << endl; 
    start = clock();
    FeatureExtractor* target_extractor = new FeatureExtractor();
    tgt_phrases->readPhraseIDsFromFile(conf["target_phraseIDs"].as<string>(), false); 
    target_extractor->readStopWords(conf["target_stopwords"].as<string>(), conf["stop_list_size"].as<int>()); 
    target_extractor->extractFeatures(tgt_phrases, conf["target_monolingual"].as<string>(), conf["window_size"].as<int>(), 1, conf["max_target_phrase_length"].as<int>()); 
    if (conf["minimum_feature_count"].as<int>() > 0)
      target_extractor->pruneFeaturesByCount(conf["minimum_feature_count"].as<int>());
    if (conf.count("analyze_feature_matrix"))
      target_extractor->analyzeFeatureMatrix(tgt_phrases->getUnlabeledPhrases());
    target_extractor->rescaleCoocToPMI();
    target_extractor->writeToFile(conf["target_feature_matrix"].as<string>(), conf["target_feature_extractor"].as<string>()); 
    cout << "Time taken: " << duration(start, clock()) << endl;     
    delete target_extractor; */
  }
  delete opts;
  delete src_phrases;
  delete tgt_phrases;
  return 0;
}
