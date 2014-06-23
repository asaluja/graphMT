#include <iostream>
#include <vector>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>
#include <time.h>
#include <omp.h>
#include "options.h"
#include "phrases.h"
#include "featext.h"
#include "graph.h"
#include "lexical.h"

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
      cout << "Time taken: " << duration(start, clock()) / numThreads << " seconds" << endl; 
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
    cout << "Time taken: " << duration(start, clock()) << " seconds" << endl; 
    start = clock();
    source_extractor->writeCoocToFile(conf["source_cooc_matrix"].as<string>()); 
    cout << "Time taken to write out co-oc file: " << duration(start, clock()) << " seconds" << endl; 
    start = clock(); 
    source_extractor->rescaleCoocToPMI();
    cout << "Time taken: " << duration(start, clock())<< " seconds" << endl; 
    start = clock(); 
    source_extractor->writeToFile(conf["source_feature_matrix"].as<string>(), conf["source_feature_extractor"].as<string>()); 
    cout << "Time taken to write out feature matrix: " << duration(start, clock()) << " seconds" << endl; 
    delete source_extractor; 
    cout << "Beginning target-side feature extraction" << endl; 
    start = clock();
    FeatureExtractor* target_extractor = new FeatureExtractor();
    tgt_phrases->readPhraseIDsFromFile(conf["target_phraseIDs"].as<string>(), false); 
    target_extractor->readStopWords(conf["target_stopwords"].as<string>(), conf["stop_list_size"].as<int>()); 
    target_extractor->extractFeatures(tgt_phrases, conf["target_monolingual"].as<string>(), conf["window_size"].as<int>(), 1, conf["max_target_phrase_length"].as<int>()); 
    if (conf["minimum_feature_count"].as<int>() > 0)
      target_extractor->pruneFeaturesByCount(conf["minimum_feature_count"].as<int>());
    if (conf.count("analyze_feature_matrix"))
      target_extractor->analyzeFeatureMatrix(tgt_phrases->getUnlabeledPhrases());
    cout << "Time taken: " << duration(start, clock()) << " seconds" << endl; 
    start = clock();
    target_extractor->writeCoocToFile(conf["target_cooc_matrix"].as<string>()); 
    cout << "Time taken to write out co-oc file: " << duration(start, clock()) << " seconds" << endl; 
    start = clock(); 
    target_extractor->rescaleCoocToPMI();
    cout << "Time taken: " << duration(start, clock())<< " seconds" << endl; 
    start = clock(); 
    target_extractor->writeToFile(conf["target_feature_matrix"].as<string>(), conf["target_feature_extractor"].as<string>()); 
    cout << "Time taken: " << duration(start, clock()) << " seconds" << endl;
    delete target_extractor; 
  }
  else if (stage == "constructgraphs"){
    FeatureExtractor* featuresFromFile = new FeatureExtractor();
    string side = conf["graph_construction_side"].as<string>();
    transform(side.begin(), side.end(), side.begin(), ::tolower);
    if (side == "source"){
      cout << "Starting graph construction on source side" << endl; 
      start = clock();
      featuresFromFile->readFromFile(conf["source_feature_matrix"].as<string>(), conf["source_feature_extractor"].as<string>()); 
      Graph* src_graph = new Graph(featuresFromFile, conf["k_nearest_neighbors"].as<int>()); 
      if (conf.count("analyze_similarity_matrix"))
	src_graph->analyzeSimilarityMatrix(src_phrases->getUnlabeledPhrases());
      cout << "Time taken: " << duration(start, clock()) << " seconds" << endl; 
      start = clock();
      src_graph->writeToFile(conf["source_similarity_matrix"].as<string>()); 
      cout << "Time taken for writing out matrix: " << duration(start, clock()) << " seconds" << endl; 
      delete src_graph; 
    }
    else if (side == "target"){
      cout << "Starting graph construction on target side" << endl; 
      start = clock(); 
      featuresFromFile->readFromFile(conf["target_feature_matrix"].as<string>(), conf["target_feature_extractor"].as<string>()); 
      Graph* tgt_graph = new Graph(featuresFromFile, conf["k_nearest_neighbors"].as<int>()); 
      if (conf.count("analyze_similarity_matrix")){
	tgt_phrases->readPhraseIDsFromFile(conf["target_phraseIDs"].as<string>(), false); //check if defined in opts
	tgt_graph->analyzeSimilarityMatrix(tgt_phrases->getUnlabeledPhrases());
      }
      cout << "Time taken: " << duration(start, clock()) << " seconds" << endl; 
      start = clock(); 
      tgt_graph->writeToFile(conf["target_similarity_matrix"].as<string>()); 
      cout << "Time taken for writing out matrix: " << duration(start, clock()) << " seconds" << endl; 
      delete tgt_graph;
    }
    else {
      cerr << "Incorrect argument for 'graph_construction_side' field" << endl; 
      exit(0);
    }    
  }
  else if (stage == "propagategraph"){
    LexicalScorer* lex = new LexicalScorer(conf["lexical_model_location"].as<string>()); 
    tgt_phrases->readPhraseIDsFromFile(conf["target_phraseIDs"].as<string>(), false); 
    src_phrases->readLabelPhraseIDsFromFile(conf["target_phraseIDs"].as<string>()); //also add to label space
    start = clock(); 
    Graph* src_graph = new Graph(conf["source_similarity_matrix"].as<string>()); 
    cout << "Time taken to read in source similarity matrix: " << duration(start, clock()) << " seconds" << endl; 
    Graph* tgt_graph = NULL; 
    string algo = conf["graph_propagation_algorithm"].as<string>();
    transform(algo.begin(), algo.end(), algo.begin(), ::tolower);
    if (conf.count("seed_target_knn") || algo == "structlabelprop"){
      start = clock(); 
      tgt_graph = new Graph(conf["target_similarity_matrix"].as<string>());     
      cout << "Time taken to read in target similarity matrix: " << duration(start, clock()) << " seconds" << endl; 
    }
    set<int> labelStopPhrases; 
    if (conf.count("filter_stop_words"))
      labelStopPhrases = FeatureExtractor::readStopWordsAsPhrases(conf["target_stopwords"].as<string>(), conf["stop_list_size"].as<int>(), tgt_phrases); 
    start = clock(); 
    src_graph->initLabelsWithLexScore(src_phrases, conf.count("seed_target_knn"), conf["mbest_processed_location"].as<string>(), lex, tgt_graph, conf["maximum_candidate_size"].as<int>(), conf.count("filter_stop_words"), labelStopPhrases);      
    cout << "Time taken to initialize unlabeled phrases' candidates: " << duration(start, clock()) << " seconds" << endl; 
    start = clock();     
    src_phrases->computeMarginals(conf["source_cooc_matrix"].as<string>()); 
    cout << "Source phrase marginals computed from co-occurrence matrix; Time taken: " << duration(start, clock()) << " seconds" << endl; 
    start = clock(); 
    tgt_phrases->computeMarginals(conf["target_cooc_matrix"].as<string>()); 
    cout << "Target phrase marginals computed from co-occurrence matrix; Time taken: " << duration(start, clock()) << " seconds" << endl; 
    cout << "Beginning label propagation" << endl; 
    start = clock();     
    for (int i = 0; i < conf["graph_propagation_iterations"].as<int>(); i++){
      if (algo == "labelprop")
	src_graph->labelProp(src_phrases); 
      else if (algo == "structlabelprop")
	src_graph->structLabelProp(src_phrases, tgt_graph); 
      else
	cerr << "Error: invalid option for graph propagation method.  Valid choices are 'LabelProp' and 'StructLabelProp'" << endl; 
      cout << "Graph Propagation iteration " << i << " complete" << endl; 
    }    
    cout << "Graph propagation complete; Time taken: " << duration(start, clock()) << " seconds" << endl; 
    src_phrases->writePhraseTable(tgt_phrases, conf["phrase_table_format"].as<string>(), conf["expanded_phrase_table_loc"].as<string>(), lex); 
    cout << "Expanded phrase table written to file" << endl; 
    delete lex; 
  }
  delete opts;
  delete src_phrases;
  delete tgt_phrases;
  return 0;
}
