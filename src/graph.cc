#include "graph.h"
#include "lexical.h"
#include <omp.h>
#include <set>

using namespace std;
using namespace Eigen;

DynamicGraph::DynamicGraph(FeatureExtractor* features){
  feat_mat = SparseMatrix<double,RowMajor>(features->getFeatureMatrix()); 
  cache = map<string, double>(); 
}

DynamicGraph::DynamicGraph(const string dgLoc){
  loadMarket(feat_mat, dgLoc); 
  cache = map<string, double>(); 
}

DynamicGraph::~DynamicGraph(){
}

void DynamicGraph::writeToFile(const string dgLoc){
  saveMarket(feat_mat, dgLoc); 
}

double DynamicGraph::getSimilarity(const int i, const int j){
  char ix[100]; 
  char iy[100]; 
  double sim; 
  sprintf(ix, "%d,%d", i, j); 
  if (cache.find(ix) == cache.end()){ //not found in cache; compute, add to cache
    SparseVector<double> vec1 = feat_mat.row(i); 
    SparseVector<double> vec2 = feat_mat.row(j); 
    sim = vec1.dot(vec2) / (vec1.norm() * vec2.norm());     
    if (sim < 0)
      cout << "Phrase ID pair (" << i << "," << j << ") has negative similarity: " << sim << endl; 
    sim = (sim < 0) ? 0 : sim; 
    cache[ix] = sim; 
    sprintf(iy, "%d,%d", j, i); 
    cache[iy] = sim; 
  }
  else
    sim = cache[ix]; 
  return sim; 
}

Graph::Graph(FeatureExtractor* features, const unsigned int k){
  sim_mat = SparseMatrix<double,RowMajor>();
  sim_mat_triplets = vector<triplet>(); 
  unsigned int featureless_phrases = 0; 
  unsigned int negative_similarities = 0; 
  #pragma omp parallel for
  for (unsigned int i = 0; i < features->getNumPoints(); i++){
    SparseVector<double> featureVec = features->getFeatureRow(i);     
    set<unsigned int> neighbors = set<unsigned int>();
    for (SparseVector<double>::InnerIterator it(featureVec); it; ++it){ //use the inverted idx structure to generate neighbors
      set<unsigned int> neighbors_by_feature = features->getNeighbors(it.index()); 
      neighbors.insert(neighbors_by_feature.begin(), neighbors_by_feature.end()); //or use set_union instead? which is faster? 
    }
    if (neighbors.size() > 0){
      vector<pair<unsigned int, double> > idxsAndDotProds = vector<pair<unsigned int, double> >(); 
      idxsAndDotProds.reserve(neighbors.size()); 
      set<unsigned int>::iterator iter; 
      for (iter = neighbors.begin(); iter != neighbors.end(); iter++){ //loop through all neighbors and compute sim
	if ((*iter) != i){ //filtering for self similarity	  
	  double dp = featureVec.dot(features->getFeatureRow(*iter)) / (featureVec.norm() * features->getFeatureRow(*iter).norm()); 
	  if (dp > 0)
	    idxsAndDotProds.push_back(make_pair(*iter, dp)); 
	}
      }
      if (idxsAndDotProds.size() > 0){ //if at least one of the similarities is positive
	sort(idxsAndDotProds.begin(), idxsAndDotProds.end(), [](const pair<unsigned int, double>& lhs, const pair<unsigned int, double>& rhs){ return lhs.second > rhs.second; }); //in descending order
	unsigned int topN = (k < idxsAndDotProds.size()) ? k : idxsAndDotProds.size(); 
	vector<pair<unsigned int, double> > topK_idxsDPs(idxsAndDotProds.begin(), idxsAndDotProds.begin()+topN);     
        #pragma omp critical(addSparseTriplet)
	{
	  for (unsigned int j = 0; j < topK_idxsDPs.size(); j++)
	    sim_mat_triplets.push_back(triplet(i, topK_idxsDPs[j].first, topK_idxsDPs[j].second)); 
	  sim_mat_triplets.push_back(triplet(i, i, 1.0)); 
	}
      }
      else { //all similarities are negative
	negative_similarities++; 
        #pragma omp critical(addSparseTripletSelfSim)
	{
	  sim_mat_triplets.push_back(triplet(i, i, 1.0)); 
	}
      }
    }
    else { //no neighbors
      featureless_phrases++; 
      #pragma omp critical(addSparseTripletSelfSimFeatureless)
	{
	  sim_mat_triplets.push_back(triplet(i, i, 1.0)); 
	}
    }    
  }
  cout << "Number of phrases without neighbors (i.e., other phrases sharing one common non stop-word feature): " << featureless_phrases << endl; 
  cout << "Number of phrases that have negative similarities with all neighbors: " << negative_similarities << endl; 
  sim_mat.resize(features->getNumPoints(), features->getNumPoints()); 
  sim_mat.reserve(sim_mat_triplets.size()); 
  sim_mat.setFromTriplets(sim_mat_triplets.begin(), sim_mat_triplets.end()); 
  sim_mat_triplets.clear(); //memory efficiency purposes
  cout << "Before symmetrizing, total NNZs in similarity matrix: " << sim_mat.nonZeros() << endl; 
  sim_mat = 0.5*(SparseMatrix<double,RowMajor>(sim_mat.transpose()) + sim_mat); 
  VectorXd indSimSumRowInv = (sim_mat*VectorXd::Ones(sim_mat.cols())).cwiseInverse(); 
  SparseMatrix<double,RowMajor> left_mult(sim_mat.rows(), sim_mat.rows());
  vector<triplet> left_mult_diagonal = vector<triplet>();
  for (unsigned int i = 0; i < indSimSumRowInv.size(); i++)
    left_mult_diagonal.push_back(triplet(i, i, indSimSumRowInv[i])); 
  left_mult.reserve(left_mult_diagonal.size());
  left_mult.setFromTriplets(left_mult_diagonal.begin(), left_mult_diagonal.end()); 
  sim_mat = left_mult * sim_mat; 
  cout << "After symmetrizing (and normalizing), total NNZs in random walk matrix: " << sim_mat.nonZeros() << endl; 
}

Graph::Graph(const string simMatLoc){
  loadMarket(sim_mat, simMatLoc); 
}


Graph::~Graph(){
}

void Graph::writeToFile(const string simMatLoc){
  saveMarket(sim_mat, simMatLoc); 
}


void Graph::analyzeSimilarityMatrix(const vector<Phrases::Phrase*> unlabeled_phrases){
  set<int> unlabeled_ids = set<int>();
  for (unsigned int i = 0; i < unlabeled_phrases.size(); i++)
    unlabeled_ids.insert(unlabeled_phrases[i]->id); 
  cout << "Dimensions: " << sim_mat.rows() << " x " << sim_mat.cols() << "; NNZs: " << sim_mat.nonZeros() << endl;
  vector<unsigned int> row_lengths = vector<unsigned int>();
  row_lengths.reserve(sim_mat.rows()); 
  int zr_lab = 0, zr_unl = 0; 
  for (int i = 0; i < sim_mat.rows(); i++){
    if (sim_mat.row(i).nonZeros() == 1){ //every phrase has a self sim      
      if (unlabeled_ids.find(i) == unlabeled_ids.end())
	zr_lab++;
      else
	zr_unl++;
    }
    row_lengths.push_back(sim_mat.row(i).nonZeros()); 
  }
  vector<unsigned int>::iterator iter = max_element(row_lengths.begin(), row_lengths.end()); 
  unsigned int idx = distance(row_lengths.begin(), iter); 
  cout << "Phrase ID " << idx << " has the most neighbors: " << *iter << endl; 
  cout << "Number of completely disconnected nodes: " << zr_lab + zr_unl << endl; 
  cout << "Number of completely disconnected labeled nodes: " << zr_lab << endl; 
  cout << "Number of completely disconnected unlabeled nodes: " << zr_unl << endl; 
}

void Graph::initLabelsWithLexScore(Phrases* src_phrases, const string mbest_processed_loc, LexicalScorer* const lex, const int maxCand_size, const bool filter_sw, set<int> stopWords){
  if (filter_sw)
    assert(stopWords.size() > 0); 
  unsigned int numCandidates = 0; 
  vector<Phrases::Phrase*> unlabeled_phrases = src_phrases->getUnlabeledPhrases(); 
  map<const string, vector<string> > mbest_by_src = src_phrases->readFormattedMBestListFromFile(mbest_processed_loc); //static method, can be read by either src_phrases or tgt_phrases
  #pragma omp parallel for
  for (unsigned int i = 0; i < unlabeled_phrases.size(); i++){ //initialize candidates for each unlabeled phrase
    const string srcphr = unlabeled_phrases[i]->phrase_str; 
     vector<string> mbest_candidates = vector<string>();
    if (mbest_by_src.find(srcphr) != mbest_by_src.end())
      mbest_candidates = mbest_by_src[srcphr]; //associate unlabeled phrase with generated candidates
    map<int,double> candidate_list = generateCandidateTranslations(srcphr, src_phrases->getPhraseID(srcphr), src_phrases, mbest_candidates, lex, maxCand_size, filter_sw, stopWords); //initialize candidate distribution
#pragma omp critical(updateDistribution)
    {
      unlabeled_phrases[i]->label_distribution.insert(candidate_list.begin(), candidate_list.end()); 
      numCandidates += unlabeled_phrases[i]->label_distribution.size(); 
    }
  }
  src_phrases->normalizeLabelDistributions(); 
  //print candidate translations? if enabled, put here
  cout << "Finished initializing candidate lists for unlabeled phrases. Average label set size: " << ((double)numCandidates)/((double)unlabeled_phrases.size()) << endl;
}

map<int, double> Graph::generateCandidateTranslations(const string phrStr, const int phrID, Phrases* const src_phrases, const vector<string> mbest_candidates, LexicalScorer* const lex, const int maxCand_size, const bool filter_sw, set<int> stopWords){
  if (filter_sw)
    assert(stopWords.size() > 0); 
  set<int> labels = set<int>(); //candidate translations
  if (sim_mat.row(phrID).nonZeros() > 1){ //make sure phrase has neighbors; > 1 because we always have self sim
    for (SparseMatrix<double,RowMajor>::InnerIterator it(sim_mat, phrID); it; ++it){
      int neighborIdx = it.col();
      if (src_phrases->getNthPhrase(neighborIdx)->isLabeled()){ //if the neighbor is labeled
	set<int> labels_from_neighbor = src_phrases->getNthPhrase(neighborIdx)->getLabels(); 
	assert(labels_from_neighbor.size() > 0); 
	labels.insert(labels_from_neighbor.begin(), labels_from_neighbor.end());
      }
    } //at this stage, we have the union of the labeled neighbors' labels
  }
  for (unsigned int i = 0; i < mbest_candidates.size(); i++){ //add generated labels
    int labelPhrID = src_phrases->getLabelPhraseID(mbest_candidates[i]); 
    if (labelPhrID > -1) //check if this is the best idea
      labels.insert(labelPhrID); 
  }
  if (filter_sw)
    filterCandidatesForStopWords(labels, stopWords); 
  vector<string> srcPhrases = vector<string>();
  vector<string> tgtPhrases = vector<string>();
  for (set<int>::iterator it = labels.begin(); it != labels.end(); it++){ //convert to phrase pairs
    srcPhrases.push_back(phrStr); 
    tgtPhrases.push_back(src_phrases->getLabelPhraseStr(*it)); 
  }
  vector<pair<double, double> > lex_scores = lex->scorePhrasePairs(srcPhrases, tgtPhrases); 
  vector<pair<int, double> > label_lexscore = vector<pair<int, double> >(); 
  for (unsigned int i = 0; i < lex_scores.size(); i++ ) //pair each candidate with score
    label_lexscore.push_back(make_pair(src_phrases->getLabelPhraseID(tgtPhrases[i]), lex_scores[i].first)); 
  if (label_lexscore.size() > 0){
    sort(label_lexscore.begin(), label_lexscore.end(), [](const pair<int, double>& lhs, const pair<int, double>& rhs){ return lhs.second > rhs.second; }); //ascending
    int topK = (maxCand_size > label_lexscore.size()) ? label_lexscore.size() : maxCand_size; 
    map<int, double> top_labels(label_lexscore.begin(), label_lexscore.begin() + topK); 
    //for (map<int, double>::iterator it = top_labels.begin(); it != top_labels.end(); it++)
    //  cout << "Label phrase ID: " << it->first << "; phrase: " << src_phrases->getLabelPhraseStr(it->first) << " has prob: " << it->second << endl;     
    return top_labels; 
  }
  else 
    return map<int,double>();
}

void Graph::filterCandidatesForStopWords(set<int>& labels, const set<int> stopWords){
  set<int> result;   
  set_difference(labels.begin(), labels.end(), stopWords.begin(), stopWords.end(), inserter(result, result.end())); 
  //if (result.size() != labels.size())
  //  cout << "Filtered stop words" << endl; 
  labels = result; 
}

void Graph::labelProp(Phrases* src_phrases){
  vector<Phrases::Phrase*> unlabeled_phrases = src_phrases->getUnlabeledPhrases(); 
  for (unsigned int i = 0; i < unlabeled_phrases.size(); i++){ 
    Phrases::Phrase* phrase = unlabeled_phrases[i]; 
    if (sim_mat.row(phrase->id).nonZeros() > 1){ //check if phrase has neighbors
      set<int> phraseLabelsIdx = phrase->getLabels(); 
      map<int,double> newLabelDistr = map<int,double>(); 
      for (SparseMatrix<double,RowMajor>::InnerIterator it(sim_mat, phrase->id); it; ++it){
	if (it.col() != phrase->id){ //filtering for self-similarity
	  set<int> neighborLabelsIdx = src_phrases->getNthPhrase(it.col())->getLabels(); 
	  vector<int> commonLabelsIdx; 
	  set_intersection(phraseLabelsIdx.begin(), phraseLabelsIdx.end(), neighborLabelsIdx.begin(), neighborLabelsIdx.end(), back_inserter(commonLabelsIdx)); 
	  if (commonLabelsIdx.size() > 0){
	    Phrases::Phrase* neighbor = src_phrases->getNthPhrase(it.col()); 
	    for (unsigned int j = 0; j < commonLabelsIdx.size(); j++){
	      int labelIdx = commonLabelsIdx[j]; 
	      double label_prob = neighbor->label_distribution[labelIdx]*sim_mat.coeff(phrase->id, neighbor->id); //guaranteed that labelIdx is in label_distribution of neighbor since it arises from the intersection
	      if (newLabelDistr.find(labelIdx) == newLabelDistr.end())
		newLabelDistr[labelIdx] = label_prob; 
	      else
		newLabelDistr[labelIdx] += label_prob;
	    } //updated label probabilities from neighbor
	  }
	}
      } //looped through all neighbors; need to normalize new labels now and transfer to label_distribution
      if (newLabelDistr.size() > 0){
	phrase->label_distribution.clear(); 
	phrase->label_distribution.insert(newLabelDistr.begin(), newLabelDistr.end()); 
	phrase->normalizeDistribution(); 
      }
    }
  }
}

void Graph::structLabelProp(Phrases* src_phrases, void* tgt_graph, bool dynamic){  
  DynamicGraph* dyn_graph = NULL; 
  Graph* graph = NULL; 
  if (dynamic)
    dyn_graph = static_cast<DynamicGraph*>(tgt_graph); 
  else
    graph = static_cast<Graph*>(tgt_graph); 
  vector<Phrases::Phrase*> unlabeled_phrases = src_phrases->getUnlabeledPhrases(); 
  for (unsigned int i = 0; i < unlabeled_phrases.size(); i++){ //loop through unlabeled phrases and update
    Phrases::Phrase* phrase = unlabeled_phrases[i]; 
    if (sim_mat.row(phrase->id).nonZeros() > 1){ //check if phrase has neighbors
      set<int> phraseLabelsIdx = phrase->getLabels(); 
      map<int,double> newLabelDistr = map<int,double>(); 
      for (SparseMatrix<double,RowMajor>::InnerIterator it(sim_mat, phrase->id); it; ++it){ //iterate through neighbors on source side
	if (it.col() != phrase->id){ //filtering for self-similarity
	  Phrases::Phrase* neighbor = src_phrases->getNthPhrase(it.col()); 
	  set<int> neighborLabelsIdx = neighbor->getLabels(); 	  
	  for (set<int>::iterator it_i = neighborLabelsIdx.begin(); it_i != neighborLabelsIdx.end(); it_i++){ //instead of computing set intersection, we loop through all neighbor labels
	    for (set<int>::iterator it_j = phraseLabelsIdx.begin(); it_j != phraseLabelsIdx.end(); it_j++){ //loop through own labels
	      double label_prob = (dynamic) ? neighbor->label_distribution[*it_i]*sim_mat.coeff(phrase->id, neighbor->id)*dyn_graph->getSimilarity(*it_j, *it_i) : neighbor->label_distribution[*it_i]*sim_mat.coeff(phrase->id, neighbor->id)*graph->getSimilarity(*it_j, *it_i); 
	      /*if (label_prob == 0){
		cout << "Zero prob. during label propagation; source phrase is '" << phrase->phrase_str << "'" << endl; 
		cout << "Candidate target phrases are '" << src_phrases->getLabelPhraseStr(*it_j) << "' (ID: " << *it_j << ") and '" << src_phrases->getLabelPhraseStr(*it_i) << "' (ID: " << *it_i << ")" << endl; 
		}*/
	      if (newLabelDistr.find(*it_j) == newLabelDistr.end()) //if label has not been added to new label distribution
		newLabelDistr[*it_j] = label_prob; 
	      else
		newLabelDistr[*it_j] += label_prob; 
	    }
	  } //for a given neighbor and given label, updated all label probabilities in own label set
	} //for a given neighbor, updated all label probabilities in own label set over all labels of neighbor
      } //went through all neighbors
      if (newLabelDistr.size() > 0){
	phrase->label_distribution.clear(); 
	phrase->label_distribution.insert(newLabelDistr.begin(), newLabelDistr.end()); 
	//can we put an if condition on the insert above? 
	//or, we loop through newLabelDistr and check if val is > 0 and if so, we add it
	phrase->normalizeDistribution(); 
      }
    }
  }      
}
