COMPILER = g++
CCFLAGS = -Wall -msse2 -O3 -fopenmp -std=c++11
INCLUDES = -I/opt/tools/boost_1_54_0/include -I. 
BOOST_PROGRAM_OPTIONS_LDFLAGS = -L/opt/tools/boost_1_54_0/lib -Wl,-rpath -Wl,/opt/tools/boost_1_54_0/lib
BOOST_PROGRAM_OPTIONS_LDPATH = /opt/tools/boost_1_54_0/lib
BOOST_PROGRAM_OPTIONS_LIBS = -lboost_program_options-mt
BOOST_IOSTREAMS_LIBS = -lboost_iostreams-mt
BOOST_FILESYSTEM_LIBS = -lboost_filesystem-mt
BOOST_SYSTEM_LIBS = -lboost_system-mt
LIBS = ${BOOST_PROGRAM_OPTIONS_LDFLAGS} ${BOOST_PROGRAM_OPTIONS_LIBS} ${BOOST_IOSTREAMS_LIBS} ${BOOST_FILESYSTEM_LIBS} ${BOOST_SYSTEM_LIBS} -lz

all: graph_prop

graph_prop: src/main.cc src/options.cc src/phrases.cc src/featext.cc
	${COMPILER} ${CCFLAGS} ${INCLUDES} -o graph_prop src/main.cc src/options.cc src/phrases.cc src/featext.cc ${LIBS}

clean:
	rm -rf *.o graph_prop
