# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude -I/sql-parser-main/src
LDFLAGS := -L/sql-parser-main/build -lsqlparser

# Directories
SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin

# Source files (add more as you expand your project)
SRCS := $(wildcard $(SRC_DIR)/*.cpp) \
        $(wildcard $(SRC_DIR)/DataHandling/*.cpp) \
        $(wildcard $(SRC_DIR)/Operations/*.cpp) \
        $(wildcard $(SRC_DIR)/QueryProcessing/*.cpp) \
        $(wildcard $(SRC_DIR)/Utilities/*.cpp) \
        $(wildcard $(SRC_DIR)/CLI/*.cpp)


# Object files
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# Main executable
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

# Main executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Pattern rule for object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Run
run: all
	./$(TARGET)

# Phony targets
.PHONY: all clean run directories