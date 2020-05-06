################################################################################
# Paths
ORTROOT=../../..

################################################################################
# Compiler
CXX = g++
CXXFLAGS = -std=c++14
INC = -I$(ORTROOT)/include/onnxruntime/core/session/
CXXFLAGS += $(INC)

################################################################################
# Linker
LIBDIR = $(ORTROOT)/build/Linux/RelWithDebInfo/
LIBS = onnxruntime
LDFLAGS = -L$(LIBDIR) -l$(LIBS)

################################################################################
# Targets
all: CXX_Api_Sample C_Api_Sample

CXX_Api_Sample: CXX_Api_Sample.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

C_Api_Sample: C_Api_Sample.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

.PHONY: clean
clean:
	rm -rf CXX_Api_Sample C_Api_Sample
