#include "options.h"
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

namespace po = boost::program_options;
using namespace std; 

Options::Options(int argc, char** argv){
  po::options_description clo("command line options");
  clo.add_options()
    ("config,c", po::value<string>(), "Configuration file")
    ("help,?", "Print this help message and exit");

  po::options_description opts("configuration options");
  opts.add_options() //list all config options here
    ("stage", po::value<string>(), "What stage to execute; values include SelectUnlabeled, CorpusSelection, FeatureExtraction, GraphConstruction, and GraphPropagation")
    ("phrase_table", po::value<string>()->default_value("-"), "Baseline phrase table location")
    ("evaluation_corpus", po::value<string>()->default_value("-"), "Location of evaluation set, from which we extract our unknown phrases that we wish to label")
    ("phrase_length", po::value<int>()->default_value(2), "Phrase length for source-side phrases (default: 2)")
    ("write_unlabeled", po::value<string>()->default_value(""), "If defined, writes out unlabeled phrases from evaluation corpus to the specified location.  Needs to be defined if 'Stage' is 'SelectUnlabeled'")
    ("analyze_unlabeled", "Categorize unlabeled phrases into all unigrams known, no unigrams known, or some unigrams known")
    ("max_target_phrase_length", po::value<int>()->default_value(5), "Maximum target phrase length to extract from monolingual corpora (default: 5)")
    ("max_target_phrase_count", po::value<int>()->default_value(100), "For each generated phrase, maximum count that we want to look for in the monolingual corpora (default: 100)")
    ("corpora_selection_side", po::value<string>()->default_value("Source"), "For corpora selection, which side we are are selecting for; values include Source and Target")    
    ("graph_construction_side", po::value<string>()->default_value("Source"), "For graph construction, which side to construct; values include Source and Target")
    ("source_stopwords", po::value<string>()->default_value(""), "Location of sorted list of source-side types, from most frequent to least frequent")
    ("target_stopwords", po::value<string>()->default_value(""), "Location of list of target-side types, from most frequent to least frequent")
    ("stop_list_size", po::value<int>()->default_value(20), "Number of frequent types to consider (default: 20)")
    ("id2feature_source", po::value<string>()->default_value(""), "Location to write the ID to feature map on the source side")
    ("id2feature_target", po::value<string>()->default_value(""), "Location to write the ID to feature map on the target side")
    ("target_phraseID", po::value<string>()->default_value(""), "Location to write the target phrase IDs to")
    ("source_monolingual", po::value<string>()->default_value(""), "Location of source monolingual corpus")
    ("target_monolingual", po::value<string>()->default_value(""), "Location of target monolingual corpus")
    ("mbest_location", po::value<string>()->default_value(""), "Location of baseline decoder m-best list")
    ("source_inverted_index", po::value<string>()->default_value(""), "Location of source inverted index data structure")
    ("target_inverted_index", po::value<string>()->default_value(""), "Location of target inverted index data structure")
    ("window_size", po::value<int>()->default_value(2), "Window size on each side to look for features for feature extraction (default: 2)")
    ("minimum_feature_count", po::value<int>()->default_value(0), "Minimum feature count of a feature for a phrase to be included in its feature space (default: 0)")
    ("k_nearest_neighbors", po::value<int>()->default_value(500), "Number of nearest neighbors to include when constructing the similarity graphs (default: 500)")
    ("source_similarity_matrix", po::value<string>()->default_value(""), "Location of source similarity matrix, in X format")
    ("target_similarity_matrix", po::value<string>()->default_value(""), "Location of target similarity matrix, in X format")
    ("seed_target_knn", "Whether to use k-nearest neighbors according to target similarity graph when seeding translation candidates for unlabeled phrases (default: false)")
    ("maximum_candidate_size", po::value<int>()->default_value(50), "Maximum number of candidates to consider for each unlabeled phrase (default: 50)");

  if (argc > 1){    
    po::store(po::parse_command_line(argc, argv, clo), conf); 
    if (conf.count("help")){
      cout <<"For config file, please define the following values in a file and pass them with the --config flag: " << endl;
      printOptions(opts);
      exit(0); 
    }
    else if (conf.count("config")){    
      string config_loc = conf["config"].as<string>();
      ifstream config_FH(config_loc.c_str());
      po::store(po::parse_config_file(config_FH, opts), conf); 
      checkParameterConsistency();
      //need to check consistency of options here
    }
    else {
      cerr << "Unrecognized argument(s)" << endl; 
      cerr << clo << endl; 
      exit(0); 
    }
  }
  else {
    cerr << clo << endl; 
    cerr << "You need to define at least a config file with the --config flag" << endl; 
    exit(0);
  }
}

Options::~Options(){  
}

po::variables_map Options::getConf(){
  return conf;
}

void Options::printOptions(po::options_description opts){
  typedef std::vector< boost::shared_ptr<po::option_description> > Ds;
  Ds const& ds=opts.options();
  for (unsigned i=0;i<ds.size();++i)
    cout<<ds[i]->long_name() << ":" << ds[i]->description() << endl;   
}

void Options::checkParameterConsistency(){
  if (!conf.count("stage")) {
    cerr << "Must define a stage in the config file" << endl; 
    exit(0);
  }
  else {
    string stage = conf["stage"].as<string>();
    transform(stage.begin(), stage.end(), stage.begin(), ::tolower);
    if (stage == "selectunlabeled"){
      if (!(conf.count("phrase_table")) || !(conf.count("write_unlabeled")) || !(conf.count("evaluation_corpus"))){
	cerr << "For 'SelectUnlabeled' stage, need to define 'phrase_table', 'write_unlabeled', and 'evaluation_corpus' fields" << endl; 
	exit(0); 
      }
    }
    else if (stage == "corpusselection"){
      cerr << "Not defined yet" << endl; 
      exit(0);
    }
    else if (stage == "featureextraction"){
      cerr << "Not defined yet" << endl; 
      exit(0);
    }
    else if (stage == "graphconstruction"){
      cerr << "Not defined yet" << endl; 
      exit(0);
    }
    else if (stage == "graphpropagation"){
      cerr << "Not defined yet" << endl; 
      exit(0);
    }
    else {
      cerr << "Invalid stage defined in config file; Please have a look at ./graph_prop --help" << endl; 
      exit(0);
    }
  }
    
}


