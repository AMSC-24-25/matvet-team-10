# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O2 -fopenmp

# Source files
SRCS = main.cpp
HEADERS = cg.hpp

# Output executable
TARGET = cg_solver

# Build target
all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Clean target
clean:
	rm -f $(TARGET)