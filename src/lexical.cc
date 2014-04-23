#include "lexical.h"
#include <iostream>
#include <Python.h>

using namespace std; 
const char* dirName = "/usr0/home/avneesh/graphMT/code/graph-propagation/scripts";
const char* libName = "/usr0/home/avneesh/tools/cdec/python/build/lib.linux-x86_64-2.7"; 
const char* scriptName = "lexical_scorer"; 
const char* funcName = "computeLexicalScores"; 
const int numArgs = 3; 

LexicalScorer::LexicalScorer(const string location){
  lexModel_loc = location; 
  assert(access(location.c_str(), F_OK) != -1); //assert for presence of lex model
  Py_Initialize(); 
  PyObject *sys = PyImport_ImportModule("sys");
  PyObject *path = PyObject_GetAttrString(sys, "path");
  PyList_Append(path, PyString_FromString(dirName)); //sys path already has libname in it from PYTHONPATH
  pModule = PyImport_ImportModule(scriptName); //should have dirNameand libName in place
  assert(pModule != NULL); 
  if (PyErr_Occurred())
    PyErr_Print();
}

LexicalScorer::~LexicalScorer(){
  Py_Finalize(); 
  Py_DECREF(pModule); //is this correct? 
  pModule = NULL; //added this
}

vector<pair<double,double> > LexicalScorer::scorePhrasePairs(vector<string> srcPhrases, vector<string> tgtPhrases){
  vector<pair<double,double> > lex_scores = vector<pair<double,double> >(); 
  PyObject *pDict, *pFunc; 
  PyObject *pArgs;
  pDict = PyModule_GetDict(pModule);  
  if (pDict != NULL){
    pFunc = PyDict_GetItemString(pDict, funcName); //borrowed function
    if (pFunc && PyCallable_Check(pFunc)){
      pArgs = PyTuple_New(numArgs); //own reference
      PyObject *model_loc = PyString_FromString(lexModel_loc.c_str()); //own reference
      PyTuple_SetItem(pArgs, 0, model_loc); //takes over ownership, no need for DECREF for model_loc
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
      if (lexScores){ //if successful, run the following
	for (unsigned int i = 0; i < PyList_Size(lexScores); i++ ){
	  PyObject *fwd_bwd_PyTuple = PyList_GetItem(lexScores, i); 
	  pair<double, double> fwd_bwd_lex = make_pair(PyFloat_AsDouble(PyTuple_GetItem(fwd_bwd_PyTuple, 0)), PyFloat_AsDouble(PyTuple_GetItem(fwd_bwd_PyTuple, 1))); 
	  lex_scores.push_back(fwd_bwd_lex); 	
	}
      }
      Py_DECREF(pArgs);       //dereference here otherwise mem leak? 
    }
    else { cerr << "Function call not successful" << endl; }
    return lex_scores; 
  }
  else { cerr << "Could not get dictionary of functions from module" << endl; }
  return lex_scores; 
}



