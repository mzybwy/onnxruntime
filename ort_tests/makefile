################################################################################
# Tests (name of cpp file will be name of exe)
TESTS = CXX_Api_Sample C_Api_Sample

################################################################################
# Paths
ORTROOT=..

################################################################################
# Compiler Config
CXX = g++
CXXFLAGS = -std=c++14
INC = -I$(ORTROOT)/include/onnxruntime/core/session/
CXXFLAGS += $(INC)

################################################################################
# Linker Config
LIBDIR = $(ORTROOT)/build/Linux/RelWithDebInfo/
LIBS = onnxruntime
LDFLAGS = -L$(LIBDIR) -l$(LIBS)

################################################################################
# Targets
all: $(TESTS)

$(TESTS): %: %.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

.PHONY: clean
clean:
	rm -rf $(TESTS)
