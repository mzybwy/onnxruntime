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
LIBDIR = -L$(ORTROOT)/build/Linux/RelWithDebInfo/ -L/usr/lib/x86_64-linux-gnu
LIBS = -lonnxruntime -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
LDFLAGS = $(LIBDIR) $(LIBS)

################################################################################
# Srcs Etc
SRCS := main.cc validator.cc
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
