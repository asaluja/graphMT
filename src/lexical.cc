#include "lexical.h"
#include <iostream>
#include <Python.h>

using namespace std; 
const string scriptName = "lexical_scorer";
const string funcName = "computeLexicalScores";
const int numArgs = 3; 

LexicalScorer::LexicalScorer(const string location){
  lexModel_loc = location; 
}

LexicalScorer::~LexicalScorer(){
}

vector<pair<double,double> > LexicalScorer::scorePhrasePairs(vector<string> srcPhrases, vector<string> tgtPhrases){
  vector<pair<double,double> > lex_scores = vector<pair<double,double> >(); 
  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs;   
  Py_Initialize();
  pName = PyString_FromString(scriptName.c_str()); 
  pModule = PyImport_Import(pName); //imports this python module
  Py_DECREF(pName);   
  if (pModule != NULL){
    pFunc = PyObject_GetAttrString(pModule, funcName.c_str()); 
    if (pFunc && PyCallable_Check(pFunc)){
      pArgs = PyTuple_New(numArgs); 
      PyObject *model_loc = PyString_FromString(lexModel_loc.c_str()); 
      PyTuple_SetItem(pArgs, 0, model_loc); 
      PyObject *srcPyObj = PyList_New(srcPhrases.size()); 
      for (unsigned int i = 0; i < srcPhrases.size(); i++)
	PyList_SetItem(srcPyObj, i, PyString_FromString(srcPhrases[i].c_str()));
      PyObject *tgtPyObj = PyList_New(tgtPhrases.size()); 
      for (unsigned int i = 0; i < tgtPhrases.size(); i++)
	PyList_SetItem(tgtPyObj, i, PyString_FromString(tgtPhrases[i].c_str())); 
      //need to have some general error tracking in case we can't convert args
      PyTuple_SetItem(pArgs, 1, srcPyObj);
      PyTuple_SetItem(pArgs, 2, tgtPyObj); 
      PyObject *lexScores = PyObject_CallObject(pFunc, pArgs); 
      for (unsigned int i = 0; i < PyList_Size(lexScores); i++ ){
	PyObject *fwd_bwd_PyTuple = PyList_GetItem(lexScores, i); 
	pair<double, double> fwd_bwd_lex = make_pair(PyFloat_AsDouble(PyTuple_GetItem(fwd_bwd_PyTuple, 0)), PyFloat_AsDouble(PyTuple_GetItem(fwd_bwd_PyTuple, 1))); 
	lex_scores.push_back(fwd_bwd_lex); 	
      }
    }
  }
  Py_Finalize(); 
  return lex_scores; 
}


