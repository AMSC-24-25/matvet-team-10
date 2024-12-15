# Compiler and flags
CXX = mpic++
STDFLAGS = -std=c++17
CPPFLAGS = -I.  -I ${mkEigenInc} -O2 -fopenmp # Include current directory for headers

# Sources and objects
SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)
HEADERS = $(wildcard *.hpp)
EXEC = $(basename $(filter main%.cpp, $(SRCS)))

# Dependencies file
DEPEND = .depend

.PHONY: all clean distclean doc

# Default target
.DEFAULT_GOAL = all

# Build all executables
all: $(EXEC)

# Rule to build each executable
$(EXEC): % : %.o $(filter-out $(basename $@).o, $(OBJS))
	$(CXX) $(STDFLAGS) $(CPPFLAGS) $^ -o $@

# Rule to build object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(STDFLAGS) $(CPPFLAGS) -c $< -o $@

# Clean targets
clean:
	rm -f $(EXEC) $(OBJS)

distclean: clean
	rm -f $(DEPEND)

# Documentation target
doc:
	doxygen $(DOXYFILE)

# Dependencies generation
$(DEPEND): $(SRCS) $(HEADERS)
	rm -f $(DEPEND)
	for f in $(SRCS); do \
		$(CXX) $(STDFLAGS) $(CPPFLAGS) -MM $$f >> $(DEPEND); \
	done

-include $(DEPEND)
