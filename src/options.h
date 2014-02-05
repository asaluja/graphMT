#pragma once

#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

namespace po = boost::program_options; 

class Options {
 private:
  po::variables_map conf;
  void printOptions(po::options_description opts);
  void checkParameterConsistency(); //internal function that makes sure if certain options are defined, then other required options are also defined

 public:
  Options(int argc, char** argv);
  ~Options();
  enum Stage { SelectUnlabeled, SelectCorpora, ExtractFeatures, ConstructGraph, PropagateGraph };
  enum GPAlgo { LabelProp, StructLabelProp }; 
  enum Side { Source, Target };
  po::variables_map getConf();
};
