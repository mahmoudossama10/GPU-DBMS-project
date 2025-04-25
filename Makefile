# ====== Paths ======
CXX := g++
NVCC := nvcc
CUDA_PATH := /usr/local/cuda
INCLUDE_DIRS := -Iinclude -Isql-parser-main/src
LIB_DIRS := -Lsql-parser-main/build -L$(CUDA_PATH)/lib64

# ====== Sources ======
CPP_SOURCES := $(shell find src -name '*.cpp')
CU_SOURCES  := $(shell find src -name '*.cu')
OBJECTS := $(CPP_SOURCES:.cpp=.o) $(CU_SOURCES:.cu=.o)
EXEC := bin/sql_processor

# ====== Flags ======
CXXFLAGS := -std=c++14 -Wall -Wextra $(INCLUDE_DIRS)
NVCCFLAGS := -std=c++14 -Xcompiler "-Wall -Wextra" $(INCLUDE_DIRS)
LDFLAGS := $(LIB_DIRS) -lsqlparser -lcudart

# Create bin directory if necessary
$(shell mkdir -p bin)

# ====== Rules ======

all: $(EXEC)

# Rule to build C++ objects
src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to build CUDA objects
src/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(EXEC): $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run: $(EXEC)
	./$(EXEC)

clean:
	rm -rf src/*.o bin/sql_processor

.PHONY: all clean run