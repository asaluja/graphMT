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
    ("stage", po::value<string>(), "What stage to execute; values include SelectUnlabeled, SelectCorpora, ExtractFeatures, ConstructGraph, PropagateGraph")
    ("number_threads", po::value<int>()->default_value(8), "Number of threads to spawn for the parallelized processes (default: 8)")
    ("phrase_table", po::value<string>()->default_value("-"), "Baseline phrase table location")
    ("phrase_table_format", po::value<string>()->default_value("cdec"), "Format of phrase table (default: cdec; accepted values: cdec, moses)")
    ("evaluation_corpus", po::value<string>()->default_value("-"), "Location of evaluation set, from which we extract our unknown phrases that we wish to label")
    ("phrase_length", po::value<int>()->default_value(2), "Phrase length for source-side phrases (default: 2)")
    ("write_unlabeled", po::value<string>()->default_value(""), "If defined, writes out unlabeled phrases from evaluation corpus to the specified location.  Needs to be defined if 'Stage' is 'SelectUnlabeled'")
    ("analyze_unlabeled", "Categorize unlabeled phrases into all unigrams known, no unigrams known, or some unigrams known")    
    ("corpora_selection_side", po::value<string>()->default_value("Source"), "For corpora selection, which side we are are selecting for; values include Source and Target")    
    ("source_mono_dir", po::value<string>()->default_value(""), "For source-side corpora selection, location of directory containing monolingual files")
    ("target_mono_dir", po::value<string>()->default_value(""), "For target-side corpora selection, location of directory containing monolingual files")
    ("source_monolingual", po::value<string>()->default_value(""), "Location of (filtered) source monolingual corpus file (not a directory)")
    ("target_monolingual", po::value<string>()->default_value(""), "Location of (filtered) target monolingual corpus file (not a directory)")
    ("max_phrase_count", po::value<int>()->default_value(100), "For each generated phrase, maximum count that we want to look for in the monolingual corpora (default: 100)")
    ("max_target_phrase_length", po::value<int>()->default_value(5), "Maximum target phrase length to extract from monolingual corpora (default: 5)")
    ("mbest_fromdecoder_location", po::value<string>()->default_value(""), "Location of m-best list produced by baseline decoder")
    ("mbest_processed_location", po::value<string>()->default_value(""), "Location of processed m-best list; written out during target-side corpora selection and read in during graph propagation when initializing translation candidates")
    ("target_phraseIDs", po::value<string>()->default_value(""), "Location to write the target phrase IDs to")
    ("source_stopwords", po::value<string>()->default_value(""), "Location of sorted list of source-side types, from most frequent to least frequent")
    ("target_stopwords", po::value<string>()->default_value(""), "Location of list of target-side types, from most frequent to least frequent")
    ("stop_list_size", po::value<int>()->default_value(20), "Number of frequent types to consider (default: 20)")
    ("source_feature_extractor", po::value<string>()->default_value(""), "Location to write inverted index and feature string to ID maps for source side")
    ("target_feature_extractor", po::value<string>()->default_value(""), "Location to write inverted index and feature string to ID maps for target side")    
    ("source_feature_matrix", po::value<string>()->default_value(""), "Location of source feature matrix")
    ("target_feature_matrix", po::value<string>()->default_value(""), "Location of target feature matrix")
    ("window_size", po::value<int>()->default_value(2), "Window size on each side to look for features for feature extraction (default: 2)")
    ("minimum_feature_count", po::value<int>()->default_value(0), "Minimum feature count of a feature for a phrase to be included in its feature space (default: 0)")
    ("analyze_feature_matrix", "Whether to analyze the feature matrices after they are constructed (default: false)")
    ("graph_construction_side", po::value<string>()->default_value("Source"), "For graph construction, which side to construct; values include Source and Target")
    ("graph_construction_method", po::value<string>()->default_value("CosineSim"), "For graph construction, which method to use (default: CosineSim)")
    ("k_nearest_neighbors", po::value<int>()->default_value(500), "Number of nearest neighbors to include when constructing the similarity graphs (default: 500)")    
    ("source_similarity_matrix", po::value<string>()->default_value(""), "Location of source similarity matrix, in X format")
    ("target_similarity_matrix", po::value<string>()->default_value(""), "Location of target similarity matrix, in X format")
    ("analyze_similarity_matrix", "Whether to analyze the similarity matrix after it is constructed (default: false)")
    ("lexical_model_location", po::value<string>()->default_value(""), "Location of lexical model, which is used when sorting translation candidates for unlabeled phrases and also as a feature value when writing out the additional phrase table")
    ("graph_propagation_algorithm", po::value<string>()->default_value("LabelProp"), "What graph propagation algorithm to use; choices include: LabelProp and StructLabelProp (default: LabelProp)")
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
      if (!(conf.count("phrase_table")) || !(conf.count("phrase_table_format")) || !(conf.count("write_unlabeled")) || !(conf.count("evaluation_corpus"))){
	cerr << "For 'SelectUnlabeled' stage, need to define 'phrase_table', 'phrase_table_format', 'write_unlabeled', and 'evaluation_corpus' fields" << endl; 
	exit(0); 
      }      
      if (conf.count("phrase_table_format")){
	string format = conf["phrase_table_format"].as<string>();
	if ((format != "cdec") && (format != "moses")){
	  cerr << "The only values supported for the 'phrase_table_format' field are 'cdec' and 'moses'" << endl; 
	  exit(0); 
	}
      }
    }
    else if (stage == "selectcorpora"){
      if (!(conf.count("corpora_selection_side")) || !(conf.count("source_mono_dir") || conf.count("target_mono_dir"))){
	cerr << "For 'SelectCorpora' stage, need to at least define the side which we are doing corpora selection on (source/target), and for that side, the monolingual directory also needs to be defined" << endl;       
	exit(0);
      }
    }  
    else if (stage == "extractfeatures"){
      if (!(conf.count("source_monolingual")) || !(conf.count("target_monolingual")) || !(conf.count("target_phraseIDs")) || !(conf.count("source_stopwords")) || !(conf.count("target_stopwords"))){
	cerr << "For 'ExtractFeatures' stage, on the input side need to define locations for source and target monolingual files (via 'source_monolingual' and 'target_monolingual' fields), the target side phrases and phrase IDs (from the corpora selection step) via 'target_phraseIDs', and the stop words on both source and target sides (via 'source_stopwords' and 'target-stopwords')" << endl; 
	exit(0);
      }
      else if (!(conf.count("source_feature_extractor")) || !(conf.count("target_feature_extractor")) || !(conf.count("source_feature_matrix")) || !(conf.count("target_feature_matrix"))){
	cerr << "For 'ExtractFeatures' stage, on the output side need to define locations for feature string to ID maps and inverted index data structures (via the 'source_feature_extractor' and 'target_feature_extractor' fields), as well as the source and target feature matrices for downstream graph construction computation (via the 'source_feature_matrix' and 'target_feature_matrix' fields)" << endl; 
	exit(0); 
      }
    }
    else if (stage == "constructgraphs"){
      if (!(conf.count("graph_construction_side")) || !(conf.count("source_feature_extractor") || conf.count("target_feature_extractor")) || !(conf.count("source_feature_matrix") || conf.count("target_feature_matrix"))){
	cerr << "For 'ConstructGraphs' stage, need to define graph construction side (source or target), and for the corresponding side, the locations of the feature matrices and inverted index/feature ID locations" << endl; 
	exit(0); 
      }
      else if (!(conf.count("source_similarity_matrix") || conf.count("target_similarity_matrix"))){
	cerr << "For 'ConstructGraphs' stage, need to define the location to write the similarity matrix" << endl; 
	exit(0); 
      }   
    }
    else if (stage == "propagategraph"){
      if (!(conf.count("source_similarity_matrix")) || !(conf.count("target_phraseIDs")) || !(conf.count("lexical_model_location"))){
	cerr << "For 'PropagateGraphs' stage, need to define at least the location of the source matrix and the target phrase IDs" << endl; 
	exit(0);
      }
      else if (conf.count("seed_target_knn") && !(conf.count("target_similarity_matrix"))){
	cerr << "For 'PropagateGraphs' stage, if 'seed_target_knn' field is enabled, then you must define 'target_similarity_matrix' field" << endl; 
	exit(0); 
      }
      string algo = conf["graph_propagation_algorithm"].as<string>();
      transform(stage.begin(), stage.end(), stage.begin(), ::tolower);
      if ((algo == "structlabelprop") && !(conf.count("target_similarity_matrix"))){
	cerr << "For 'PropagateGraphs' stage, if 'StructLabelProp' is the graph propagation algorithm, then you must define 'target_similarity_matrix' field" << endl; 
	exit(0);
      }
    }
    else {
      cerr << "Invalid stage defined in config file; Please have a look at ./graph_prop --help" << endl; 
      exit(0);
    }
  }
    
}


