`graphMT` is a package for graph-based machine translation.  It is a direct implementation of the following paper:

Graph-based Semi-Supervised Learning of Translation Models from Monolingual Data.
Avneesh Saluja, Hany Hassan, Kristina Toutanova, and Chris Quirk.
ACL 2014.

## System requirements

To be honest, the current system has not been tested in a wide variety of environments. If you find compilation issues in other environments, please let me know.  

- A Linux or Mac OS X system (tested on Ubuntu 12.04)
- A C++ compiler implementing the [C++-11 standard](http://www.stroustrup.com/C++11FAQ.html)
    - either with the C++-0x flag or C++11 flag, depending on the version of the compiler
    - this code has been compiled with GCC versions 4.5, 4.6, and 4.7
- [Boost C++ libraries (version 1.54 or later)](http://www.boost.org/)
  	 - If you build your own boost, you _must install it_ using `bjam install`
	 - Older versions of Boost _may_ work
- [Eigen C++ libraries](http://eigen.tuxfamily.org/)

## Building

- Edit the following fields in the Makefile:
  - `CCFLAGS`: either `-std=c++11` or `-std=c++0x` depending on your compiler
  - `EIGENPATH`: where your Eigen header files are
  - `INCLUDES`: where the Boost and Python header files are
  - `BOOST_LDFLAGS`: where the Boost .so files are
- run `make` in the root directory

## End-to-end Instructions

The pipeline is controlled by a series of changes in the configuration file. Sample configuration files have been provided.  
There are 5 stages to the pipeline: `SelectUnlabeled`, `SelectCorpora`, `ExtractFeatures`, `ConstructGraphs`, and `PropagateGraph`. 

#### Prerequisites:
- A tuned baseline system with the accompanying parallel corpora and development set. 
- Additional monolingual corpora in both source and target languages. 
- An evaluation set. 

#### Steps:
- Combine the development set and evaluation set.  Modify the appropriate fields in config file and run `./graph_prop -c config.ini` (see `select_unlabeled.ini`)
- Run baseline system on unlabeled phrases, and write out m-best phrases (e.g., for `moses`, run with the `-n-best-list` flag on)
- Split monolingual corpora into evenly-sized files for corpus selection:

  ```
  split -l numLinesPerFile -a 3 -d file
  ```

  - Then, write a simple bash for loop to move/rename the files in the format "hi.X.gz" or "en.X.gz" where X is the file number. Don't forget that each file should be gzipped!
- Run the SelectCorpora step on the source and target sides (see `cs.src.ini` and `cs.tgt.ini`)
- After obtaining the selected monolingual corpora on both sides, concatenate with the parallel corpora. 
- Create the stop-word list for both source and target languages using the concatenated monolingual + parallel corpora.  For example, we can use SRILM (or other tools):

  ```
  ngram-count -text monolingual.text -write-order 1 -write en.1cnt;
  sort -k2rn en.1cnt > en.1cnt.sorted
  ```

- Run the feature extraction step (see `extract_features.ini`)
- Run the graph construction steps on both sides (see `gc.src.ini` and `gc.tgt.ini`)
- Run the graph propagation step (see `propagate_graphs.ini`)
  - Note that this step requires a lexical model as input.  Currently, there is support for the suffix array-based lexical models extracted using `Pycdec` as part of the default phrasal extraction process in cdec.  Support needs to be extended for other lexical model formats. 

## Things to add

- full support for grammars/phrase tables extracted from cdec's phrase extraction process
- integration with pre-computed word or phrase representations for on-the-fly target similarity computation (obviating the need for a target graph)
- re-estimation of target phrase distributions for labeled source phrases (i.e., from the phrase table), leading towards applications in domain adaptation
- support for different similarity computation techniques
- support for neural-based distributed representations for words/phrases
- multiple n-gram order handling (e.g., unigrams and bigrams together)
- integration with [FLANN](http://www.cs.ubc.ca/research/flann/) for fast computation of similarity graphs
- support for other lexical model formats (e.g., GIZA++)
- compilation with autoconf/automake tools

## Citation

If you make use of this package, please cite:

A. Saluja, H. Hassan, K. Toutanova, and C. Quirk. Graph-based Semi-Supervised Learning of Translation Models from Monolingual Data. In *Proceedings of ACL*, June, 2014. [[bibtex](http://aclweb.org/anthology/P/P14/P14-1064.bib)] [[pdf](http://aclweb.org/anthology/P/P14/P14-1064.pdf)]


