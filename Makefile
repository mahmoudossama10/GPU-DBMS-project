# Compilers
CXX := g++
NVCC := nvcc

# Flags
CXXFLAGS := -std=c++14 -Wall -Wextra -Iinclude -I/sql-parser-main/src
NVCCFLAGS := -std=c++14 -Iinclude -I/sql-parser-main/src
LDFLAGS := -L/sql-parser-main/build -lsqlparser -lcudart -L/usr/local/cuda/lib64

# Directories
SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin

# Source files
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp) \
            $(wildcard $(SRC_DIR)/DataHandling/*.cpp) \
            $(wildcard $(SRC_DIR)/Operations/*.cpp) \
            $(wildcard $(SRC_DIR)/QueryProcessing/*.cpp) \
            $(wildcard $(SRC_DIR)/Utilities/*.cpp) \
            $(wildcard $(SRC_DIR)/CLI/*.cpp)

CU_SRCS := $(wildcard $(SRC_DIR)/QueryProcessing/*.cu)

# Object files
CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))

OBJS := $(CPP_OBJS) $(CU_OBJS)

# Final executable
TARGET := $(BIN_DIR)/sql_processor

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)/DataHandling
	@mkdir -p $(BUILD_DIR)/Operations
	@mkdir -p $(BUILD_DIR)/QueryProcessing
	@mkdir -p $(BUILD_DIR)/Utilities
	@mkdir -p $(BUILD_DIR)/CLI
	@mkdir -p $(BIN_DIR)

# Link final binary
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Compile .cpp files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile .cu files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean build
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Run binary
run: all
	./$(TARGET)

# Mark targets as phony
.PHONY: all clean run directories
