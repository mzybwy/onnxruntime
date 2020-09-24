################################################################################
# Paths
ORTROOT=../../..

################################################################################
# Compiler Config
CXX = g++
CXXFLAGS = -std=c++14
INC = -I$(ORTROOT)/include -I$(ORTROOT)/include/onnxruntime/core/session/
CXXFLAGS += $(INC)

################################################################################
# Linker Config
LIBDIR = $(ORTROOT)/build/Linux/RelWithDebInfo/
LIBS = onnxruntime
LDFLAGS = -L$(LIBDIR) -l$(LIBS)

################################################################################
# Srcs Etc
SRCS := main.cc
OBJS := $(SRCS:%=%.o)


################################################################################
# Targets
all: exe

exe: $(OBJS)
	$(CXX) $(OBJS) -o $@ $(CXXFLAGS) $(LDFLAGS)

# c++ source
%.cc.o: %.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


.PHONY: clean
clean:
	rm -rf exe *.o
