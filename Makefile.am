AUTOMAKE_OPTIONS = foreign
bin_PROGRAMS = graph_propagation
graph_propagation_SOURCES = src/options.h src/options.cc src/main.cc
graph_propagation_LDFLAGS = -lz
AM_CPPFLAGS = -Wall -march=native -mtune=native -msse2 -fopenmp -O3 -pipe -I/usr/include/boost -I.
