#include "../../include/QueryProcessing/GPU.hpp"
#include "../../include/Utilities/ErrorHandling.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <chrono>
// CUDA kernels
using namespace std::chrono;

// Helper for string comparison on device
__device__ int strcmp_device(const char* str1, const char* str2) {
    while (*str1 && (*str1 == *str2)) {
        str1++;
        str2++;
    }
    return *(const unsigned char*)str1 - *(const unsigned char*)str2;
}


__device__ unsigned int matchCounter = 0;

// Kernel to convert boolean matrix to array of index pairs
__global__ void convertToPairs(
    const uint8_t* resultMatrix,
    int2* pairs,
    int leftSize,
    int rightSize,
    int maxPairs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = leftSize * rightSize;
    
    if (idx < totalSize) {
        int i = idx / rightSize;  // Row index (left table)
        int j = idx % rightSize;  // Column index (right table)
        
        if (resultMatrix[idx] == 1) {
            // Found a match, add it to pairs array
            unsigned int pairIdx = atomicAdd(&matchCounter, 1);
            if (pairIdx < maxPairs) {
                pairs[pairIdx] = make_int2(i, j);
            }
        }
    }
}

// Reset counter kernel
__global__ void resetMatchCounter() {
    matchCounter = 0;
}

// Get final counter value kernel
__global__ void getMatchCount(unsigned int* count) {
    *count = matchCounter;
}



__global__ void compareStringColumns(
     char** leftColumn, 
     char** rightColumn,
    int leftSize, 
    int rightSize,
    uint8_t* results, 
    int opType) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < leftSize && j < rightSize) {
        uint8_t match = 0;
        
        switch (opType) {
            case 0: // Equals
                match = (strcmp_device(leftColumn[i], rightColumn[j]) == 0) ? 1 : 0;
                break;
            case 1: // Not Equals
                match = (strcmp_device(leftColumn[i], rightColumn[j]) != 0) ? 1 : 0;
                break;
            case 2: // Less Than
                match = (strcmp_device(leftColumn[i], rightColumn[j]) < 0) ? 1 : 0;
                break;
            case 3: // Greater Than
                match = (strcmp_device(leftColumn[i], rightColumn[j]) > 0) ? 1 : 0;
                break;
            case 4: // Less Than or Equals
                match = (strcmp_device(leftColumn[i], rightColumn[j]) <= 0) ? 1 : 0;
                break;
            case 5: // Greater Than or Equals
                match = (strcmp_device(leftColumn[i], rightColumn[j]) >= 0) ? 1 : 0;
                break;
        }
        
        results[i * rightSize + j] = match;
    }
}



__global__ void compareStringColumnsOneTable(
    char** leftColumn, 
    char** rightColumn,
   int leftSize, 
   int rightSize,
   uint8_t* results, 
   int opType) 
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (i < leftSize) {
       uint8_t match = 0;
       
       switch (opType) {
           case 0: // Equals
               match = (strcmp_device(leftColumn[i], rightColumn[i]) == 0) ? 1 : 0;
               break;
           case 1: // Not Equals
               match = (strcmp_device(leftColumn[i], rightColumn[i]) != 0) ? 1 : 0;
               break;
           case 2: // Less Than
               match = (strcmp_device(leftColumn[i], rightColumn[i]) < 0) ? 1 : 0;
               break;
           case 3: // Greater Than
               match = (strcmp_device(leftColumn[i], rightColumn[i]) > 0) ? 1 : 0;
               break;
           case 4: // Less Than or Equals
               match = (strcmp_device(leftColumn[i], rightColumn[i]) <= 0) ? 1 : 0;
               break;
           case 5: // Greater Than or Equals
               match = (strcmp_device(leftColumn[i], rightColumn[i]) >= 0) ? 1 : 0;
               break;
       }
       
       results[i] = match;
   }
}


__global__ void compareIntColumns(
     int* leftColumn, 
     int* rightColumn,
    int leftSize, 
    int rightSize,
    uint8_t* results, 
    int opType) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < leftSize && j < rightSize) {
        uint8_t match = 0;
        
        switch (opType) {
            case 0: // Equals
                match = (leftColumn[i] == rightColumn[j]) ? 1 : 0;
                break;
            case 1: // Not Equals
                match = (leftColumn[i] != rightColumn[j]) ? 1 : 0;
                break;
            case 2: // Less Than
                match = (leftColumn[i] < rightColumn[j]) ? 1 : 0;
                break;
            case 3: // Greater Than
                match = (leftColumn[i] > rightColumn[j]) ? 1 : 0;
                break;
            case 4: // Less Than or Equals
                match = (leftColumn[i] <= rightColumn[j]) ? 1 : 0;
                break;
            case 5: // Greater Than or Equals
                match = (leftColumn[i] >= rightColumn[j]) ? 1 : 0;
                break;
        }
        
        results[i * rightSize + j] = match;
    }
}


__global__ void compareIntColumnsOmeTable(
    int* leftColumn, 
    int* rightColumn,
   int leftSize, 
   int rightSize,
   uint8_t* results, 
   int opType) 
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (i < leftSize) {
       uint8_t match = 0;
       
       switch (opType) {
           case 0: // Equals
               match = (leftColumn[i] == rightColumn[i]) ? 1 : 0;
               break;
           case 1: // Not Equals
               match = (leftColumn[i] != rightColumn[i]) ? 1 : 0;
               break;
           case 2: // Less Than
               match = (leftColumn[i] < rightColumn[i]) ? 1 : 0;
               break;
           case 3: // Greater Than
               match = (leftColumn[i] > rightColumn[i]) ? 1 : 0;
               break;
           case 4: // Less Than or Equals
               match = (leftColumn[i] <= rightColumn[i]) ? 1 : 0;
               break;
           case 5: // Greater Than or Equals
               match = (leftColumn[i] >= rightColumn[i]) ? 1 : 0;
               break;
       }
       
       results[i] = match;
   }
}

__global__ void compareIntWithConstant(
     int* column, 
    int constant,
    int size, 
    uint8_t* results, 
    int opType) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        uint8_t match = 0;
        
        switch (opType) {
            case 0: // Equals
                match = (column[i] == constant) ? 1 : 0;
                break;
            case 1: // Not Equals
                match = (column[i] != constant) ? 1 : 0;
                break;
            case 2: // Less Than
                match = (column[i] < constant) ? 1 : 0;
                break;
            case 3: // Greater Than
                match = (column[i] > constant) ? 1 : 0;
                break;
            case 4: // Less Than or Equals
                match = (column[i] <= constant) ? 1 : 0;
                break;
            case 5: // Greater Than or Equals
                match = (column[i] >= constant) ? 1 : 0;
                break;
        }
        
        results[i] = match;
    }
}
// New kernel that uses the improved string storage approach
__global__ void compareStringWithConstantImproved(
    const char* stringBuffer,
    const size_t* stringOffsets,
    const char* constant,
    int size,
    uint8_t* results,
    int opType)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        // Get pointer to the current string using the offset
        const char* currentString = stringBuffer + stringOffsets[i];
        
        int comparison = 0;
        // Compare strings
        while (*currentString == *constant && *currentString && *constant) {
            currentString++;
            constant++;
        }
        
        comparison = (unsigned char)*currentString - (unsigned char)*constant;
        
        uint8_t match = 0;
        switch (opType) {
            case 0: // Equals
                match = (comparison == 0) ? 1 : 0;
                break;
            case 1: // Not Equals
                match = (comparison != 0) ? 1 : 0;
                break;
            case 2: // Less Than
                match = (comparison < 0) ? 1 : 0;
                break;
            case 3: // Greater Than
                match = (comparison > 0) ? 1 : 0;
                break;
            case 4: // Less Than or Equals
                match = (comparison <= 0) ? 1 : 0;
                break;
            case 5: // Greater Than or Equals
                match = (comparison >= 0) ? 1 : 0;
                break;
        }
        
        results[i] = match;
    }
}



__global__ void combineResults(
    const uint8_t* results1, 
    const uint8_t* results2,
    uint8_t* output, 
    int size, 
    uint8_t isAnd) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        bool andResult = results1[i] && results2[i];
        bool orResult = results1[i] || results2[i];
        output[i] = (isAnd * andResult) | ((!isAnd) * orResult);
    }
}

GPUManager::GPUManager() {
    // Check if CUDA is available
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        std::cout << "No CUDA-capable GPU found. Using CPU processing." << std::endl;
        hasGPU_ = false;
    } else {
        std::cout << "GPU acceleration available. Found " << deviceCount << " CUDA device(s)." << std::endl;
        hasGPU_ = true;
    }
}

// Helper function to check if a string can be parsed as an integer
bool isInteger(const std::string& str) {
    if (str.empty()) return false;
    
    size_t start = 0;
    if (str[0] == '-' || str[0] == '+') {
        if (str.size() == 1) return false;
        start = 1;
    }
    
    for (size_t i = start; i < str.size(); i++) {
        if (!std::isdigit(str[i])) return false;
    }
    
    return true;
}

GPUManager::~GPUManager() {
    // Clean up any GPU resources if needed
}

bool GPUManager::isGPUAvailable() const {
    return hasGPU_;
}

int GPUManager::findColumnIndex(const Table& table, const char* columnName, const char* tableName) {
    const auto& headers = table.getHeaders();
    
    for (size_t i = 0; i < headers.size(); i++) {
        // If table name is specified, check for "tableName.columnName" format
        if (tableName) {
            std::string fullColumnName = std::string(tableName) + "." + std::string(columnName);
            if (headers[i] == fullColumnName || headers[i] == std::string(columnName)) {
                return static_cast<int>(i);
            }
        } 
        // Otherwise check for just the column name
        else if (headers[i] == columnName) {
            return static_cast<int>(i);
        }
    }
    
    return -1; // Column not found
}
std::pair<std::vector<std::pair<int, int>>, int> GPUManager::gpuJoinTables(
    const Table& leftTable, 
    const Table& rightTable,
    const hsql::Expr* conditions) 
{
    if (!hasGPU_) {
        throw std::runtime_error("GPU operations not available");
    }
    
    int leftSize = leftTable.getSize();
    int rightSize = rightTable.getSize();
    int resultSize = leftSize * rightSize;
    
    std::vector<uint8_t> resultMatrix(resultSize, 0);
    
    // Process each condition and combine results
    if (conditions->type == hsql::kExprOperator) {
        if (conditions->opType == hsql::OperatorType::kOpAnd || conditions->opType == hsql::OperatorType::kOpOr) {
            // Process binary kOpAnd/kOpOr operations
            auto leftResults = processBinaryExpr(leftTable, rightTable, conditions->expr);
            auto rightResults = processBinaryExpr(leftTable, rightTable, conditions->expr2);
            
            // Create device vectors
            uint8_t *d_leftResults, *d_rightResults, *d_output;
            cudaMalloc(&d_leftResults, resultSize * sizeof(uint8_t));
            cudaMalloc(&d_rightResults, resultSize * sizeof(uint8_t));
            cudaMalloc(&d_output, resultSize * sizeof(uint8_t));
            
            // Copy data to device
            cudaMemcpy(d_leftResults, leftResults.data(), resultSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rightResults, rightResults.data(), resultSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
            
            // Set up kernel execution parameters
            int blockSize = 256;
            int numBlocks = (resultSize + blockSize - 1) / blockSize;
            
            // Execute kernel
            uint8_t isAnd = conditions->opType == hsql::OperatorType::kOpAnd ? 1 : 0;
            combineResults<<<numBlocks, blockSize>>>(d_leftResults, d_rightResults, d_output, resultSize, isAnd);
            
            // Copy results back to host
            cudaMemcpy(resultMatrix.data(), d_output, resultSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            
            // Free device memory
            cudaFree(d_leftResults);
            cudaFree(d_rightResults);
            cudaFree(d_output);
        } 
        else {
            // Process comparison operation
            resultMatrix = processComparisonExpr(leftTable, rightTable, conditions);
        }
    }
    
    // Now convert the result matrix to pairs
    
    // Count the number of 1s in the matrix (for CPU fallback and sizing)
    int matchCount = 0;
    for (int i = 0; i < resultSize; i++) {
        if (resultMatrix[i] == 1) {
            matchCount++;
        }
    }
    
    // Prepare to convert result matrix to pairs
    int maxPairs = matchCount;  // We know exactly how many pairs we need
    
    // Allocate device memory
    uint8_t* d_resultMatrix;
    int2* d_resultPairs;
    unsigned int* d_count;
    
    cudaMalloc(&d_resultMatrix, resultSize * sizeof(uint8_t));
    cudaMalloc(&d_resultPairs, maxPairs * sizeof(int2));
    cudaMalloc(&d_count, sizeof(unsigned int));
    
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Copy result matrix to device
    cudaMemcpy(d_resultMatrix, resultMatrix.data(), resultSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    // Reset counter
    resetMatchCounter<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    // Convert to pairs
    int blockSize = 256;
    int numBlocks = (resultSize + blockSize - 1) / blockSize;
    convertToPairs<<<numBlocks, blockSize>>>(d_resultMatrix, d_resultPairs, leftSize, rightSize, maxPairs);
    cudaDeviceSynchronize();
    
    // Get actual number of pairs found
    unsigned int actualMatchCount;
    getMatchCount<<<1, 1>>>(d_count);
    cudaMemcpy(&actualMatchCount, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
    // Copy pairs back to host
    std::vector<int2> tempPairs(matchCount);
    cudaMemcpy(tempPairs.data(), d_resultPairs, matchCount * sizeof(int2), cudaMemcpyDeviceToHost);
    
    // Convert int2 to std::pair<int, int>
    std::vector<std::pair<int, int>> resultPairs(matchCount);
    for (int i = 0; i < matchCount; i++) {
        resultPairs[i] = std::make_pair(tempPairs[i].x, tempPairs[i].y);
    }
    


    
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "pairs compareIntColumns time: " << elapsedTime << " milliseconds" << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    // Free device memory
    cudaFree(d_resultMatrix);
    cudaFree(d_resultPairs);
    cudaFree(d_count);
    
    return {resultPairs, matchCount};
}

std::shared_ptr<Table> GPUManager::executeJoin(std::shared_ptr<Table> leftTable,
    std::shared_ptr<Table> rightTable,
    const hsql::Expr* condition) {
    // Get GPU join pairs and count
    auto result = gpuJoinTables(*leftTable, *rightTable, condition);
    auto joinPairs = result.first;
    auto matchCount = result.second;

    // Create result table structure
    auto headers = combineHeaders(*leftTable, *rightTable);
    auto data = mergeJoinResults(*leftTable, *rightTable, joinPairs);

    return std::make_shared<Table>(
    leftTable->getName() + "_joined_" + rightTable->getName(),
    headers,
    data
);
}

std::shared_ptr<Table> GPUManager::applyFilter(const Table& table, 
                                             const std::vector<uint8_t>& mask) 
{
    return std::make_shared<Table>(
        table.getName() + "_filtered",
        table.getHeaders(),
        mergeFilterResults(table, mask)
    );
}

std::vector<std::vector<std::string>> GPUManager::mergeFilterResults(
    const Table& table,
    const std::vector<uint8_t>& mask) const 
{
    std::vector<std::vector<std::string>> result;
    
    #pragma omp parallel for
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) {
            #pragma omp critical
            result.push_back(table.getRow(i));
        }
    }
    
    return result;
}

std::vector<std::string> GPUManager::combineHeaders(const Table& left,
                                                  const Table& right) const {
    std::vector<std::string> headers;
    
    // Add left headers with alias
    for (const auto& h : left.getHeaders()) {
        headers.push_back(left.getAlias().empty() ? h : left.getAlias() + "." + h);
    }
    
    // Add right headers with alias
    for (const auto& h : right.getHeaders()) {
        headers.push_back(right.getAlias().empty() ? h : right.getAlias() + "." + h);
    }
    
    return headers;
}

std::vector<std::vector<std::string>> GPUManager::mergeJoinResults(
    const Table& left,
    const Table& right,
    const std::vector<std::pair<int, int>>& joinPairs) const {
    
    std::vector<std::vector<std::string>> result;
    result.reserve(joinPairs.size()); // Pre-allocate for better performance
    
    auto start = high_resolution_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < joinPairs.size(); ++i) {
        const auto& pair = joinPairs[i];
        int leftIdx = pair.first;
        int rightIdx = pair.second;

        
        // Combine rows
        auto combined = left.getRow(leftIdx);
        const auto& rightRow = right.getRow(rightIdx);
        combined.insert(combined.end(), rightRow.begin(), rightRow.end());
        
        #pragma omp critical
        result.push_back(std::move(combined));
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "Join merge time: " << joinPairs.size() << " matches, " << duration.count() << " ms" << std::endl;
    
    return result;
}

std::vector<uint8_t> GPUManager::gpuFilterTable(
    const Table& table,
    const hsql::Expr* conditions) 
{
    if (!hasGPU_) {
        throw std::runtime_error("GPU operations not available");
    }

    
    int tableSize = table.getSize();
    std::vector<uint8_t> resultVector(tableSize, 0);
    
    // Process the conditions (simplified for the example)
    if (conditions->type == hsql::kExprOperator) {
        // Handle comparison with constant
        if (conditions->expr->type == hsql::kExprColumnRef && 
            (conditions->expr2->type == hsql::kExprLiteralInt || 
             conditions->expr2->type == hsql::kExprLiteralString)) {
            
            const char* columnName = conditions->expr->name;
            int columnIndex = findColumnIndex(table, columnName, conditions->expr->table);
            
            if (columnIndex == -1) {
                throw std::runtime_error("Column not found: " + std::string(columnName));
            }
            
            uint8_t* d_results;
            cudaMalloc(&d_results, tableSize * sizeof(uint8_t));
            cudaMemset(d_results, 0, tableSize * sizeof(uint8_t));
            
            int blockSize = 256;
            int numBlocks = (tableSize + blockSize - 1) / blockSize;
            
            // Convert operator type to our internal representation
            int opType;
            switch (conditions->opType) {
                case hsql::OperatorType::kOpEquals: opType = 0; break;
                case hsql::OperatorType::kOpNotEquals: opType = 1; break;
                case hsql::OperatorType::kOpLess: opType = 2; break;
                case hsql::OperatorType::kOpGreater: opType = 3; break;
                case hsql::OperatorType::kOpLessEq: opType = 4; break;
                case hsql::OperatorType::kOpGreaterEq: opType = 5; break;
                default: throw std::runtime_error("Unsupported operator type");
            }
            
            // For integer comparison
            if (conditions->expr2->type == hsql::kExprLiteralInt) {
                int constant = conditions->expr2->ival;
                
                // Prepare column data for GPU
                std::vector<int> columnData(tableSize);
                const auto& data = table.getData();
                for (int i = 0; i < tableSize; i++) {
                    columnData[i] = std::stoi(data[i][columnIndex]);
                }
                
                int* d_column;
                cudaMalloc(&d_column, tableSize * sizeof(int));
                cudaMemcpy(d_column, columnData.data(), tableSize * sizeof(int), cudaMemcpyHostToDevice);
                
                compareIntWithConstant<<<numBlocks, blockSize>>>(d_column, constant, tableSize, d_results, opType);
                cudaFree(d_column);
            }
            // For string comparison
            else if (conditions->expr2->type == hsql::kExprLiteralString) {
                const char* constant = conditions->expr2->name;
                size_t constantLen = strlen(constant) + 1; // +1 for null terminator
                
                // Get string data from the table
                const auto& data = table.getData();
                
                // Calculate total buffer size needed for all strings
                size_t totalBufferSize = 0;
                std::vector<size_t> stringLengths(tableSize);
                std::vector<size_t> stringOffsets(tableSize);
                
                for (int i = 0; i < tableSize; i++) {
                    stringLengths[i] = data[i][columnIndex].length() + 1; // +1 for null terminator
                    stringOffsets[i] = totalBufferSize;
                    totalBufferSize += stringLengths[i];
                }
                
                // Create a buffer for all strings
                char* stringBuffer = new char[totalBufferSize];
                
                // Copy strings to the buffer
                for (int i = 0; i < tableSize; i++) {
                    strcpy(stringBuffer + stringOffsets[i], data[i][columnIndex].c_str());
                }
                
                // Allocate device memory
                char* d_stringBuffer;
                size_t* d_stringOffsets;
                char* d_constant;
                
                cudaMalloc(&d_stringBuffer, totalBufferSize);
                cudaMalloc(&d_stringOffsets, tableSize * sizeof(size_t));
                cudaMalloc(&d_constant, constantLen);
                
                // Copy data to device
                cudaMemcpy(d_stringBuffer, stringBuffer, totalBufferSize, cudaMemcpyHostToDevice);
                cudaMemcpy(d_stringOffsets, stringOffsets.data(), tableSize * sizeof(size_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_constant, constant, constantLen, cudaMemcpyHostToDevice);
                
                // Launch kernel with improved string handling
                compareStringWithConstantImproved<<<numBlocks, blockSize>>>(
                    d_stringBuffer, d_stringOffsets, d_constant, tableSize, d_results, opType);
                
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                }
                
                // Free allocated memory
                delete[] stringBuffer;
                cudaFree(d_stringBuffer);
                cudaFree(d_stringOffsets);
                cudaFree(d_constant);
            }
            
            // Copy results back
            cudaMemcpy(resultVector.data(), d_results, tableSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            cudaFree(d_results);
        }


        else{
            // Column-column comparison
            if (conditions->expr->type == hsql::kExprColumnRef && conditions->expr2->type == hsql::kExprColumnRef) {
              const char* leftColName = conditions->expr->name;
              const char* rightColName = conditions->expr2->name;
              
              int leftColIndex = findColumnIndex(table, leftColName, conditions->expr->table);
              int rightColIndex = findColumnIndex(table, rightColName, conditions->expr2->table);
              
              
              // Determine the operator type
              int opType;
              switch (conditions->opType) {
                  case hsql::OperatorType::kOpEquals: opType = 0; break;
                  case hsql::OperatorType::kOpNotEquals: opType = 1; break;
                  case hsql::OperatorType::kOpLess: opType = 2; break;
                  case hsql::OperatorType::kOpGreater: opType = 3; break;
                  case hsql::OperatorType::kOpLessEq: opType = 4; break;
                  case hsql::OperatorType::kOpGreaterEq: opType = 5; break;
                  default: throw std::runtime_error("Unsupported operator type");
              }
              
              const auto& leftData = table.getData();
              const auto& rightData = table.getData();
              
              // Check if we're dealing with integer or string columns
              bool isIntegerComparison = false;
              if (!leftData.empty() && !rightData.empty()) {
                  // Sample the first row of each table to determine type
                  isIntegerComparison = isInteger(leftData[0][leftColIndex]) && isInteger(rightData[0][rightColIndex]);
              }
              
              // Set up grid and block dimensions for 2D execution
              dim3 blockDim(16, 16);
              dim3 gridDim(
                  (tableSize + blockDim.x - 1) / blockDim.x,
                  (tableSize + blockDim.y - 1) / blockDim.y
              );
              
              if (isIntegerComparison) {
                  // Handle integer columns
                  std::vector<int> leftColData(tableSize);
                  std::vector<int> rightColData(tableSize);
                  
                  // Prepare column data
                  for (int i = 0; i < tableSize; i++) {
                      leftColData[i] = std::stoi(leftData[i][leftColIndex]);
                  }
                  
                  for (int i = 0; i < tableSize; i++) {
                      rightColData[i] = std::stoi(rightData[i][rightColIndex]);
                  }
                  
                  // Allocate device memory
                  int *d_leftCol, *d_rightCol;
                  uint8_t *d_results;
                  
                  cudaMalloc(&d_leftCol, tableSize * sizeof(int));
                  cudaMalloc(&d_rightCol, tableSize * sizeof(int));
                  cudaMalloc(&d_results, tableSize * sizeof(uint8_t));
                  
                  // Copy data to device
                  cudaMemcpy(d_leftCol, leftColData.data(), tableSize * sizeof(int), cudaMemcpyHostToDevice);
                  cudaMemcpy(d_rightCol, rightColData.data(), tableSize * sizeof(int), cudaMemcpyHostToDevice);
                  
                  cudaEvent_t start, stop;
                  cudaEventCreate(&start);
                  cudaEventCreate(&stop);
                  cudaEventRecord(start);
                  // Launch integer comparison kernel
                  compareIntColumnsOmeTable<<<gridDim, blockDim>>>(
                      d_leftCol, d_rightCol, tableSize, tableSize, d_results, opType);
                  
                      cudaEventRecord(stop);
                      cudaEventSynchronize(stop);
                      float elapsedTime;
                      cudaEventElapsedTime(&elapsedTime, start, stop);
                      std::cout << "Elapsed compareIntColumns time: " << elapsedTime << " milliseconds" << std::endl;
                      cudaEventDestroy(start);
                      cudaEventDestroy(stop);
                      
                  // Copy results back to host
                  cudaMemcpy(resultVector.data(), d_results, tableSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                  
                  // Free device memory
                  cudaFree(d_leftCol);
                  cudaFree(d_rightCol);
                  cudaFree(d_results);
              } else {
                  // Handle string columns
                  std::vector<std::string> leftColData(tableSize);
                  std::vector<std::string> rightColData(tableSize);
                  
                  // Prepare column data
                  for (int i = 0; i < tableSize; i++) {
                      leftColData[i] = leftData[i][leftColIndex];
                  }
                  
                  for (int i = 0; i < tableSize; i++) {
                      rightColData[i] = rightData[i][rightColIndex];
                  }
                  
                  // Create array of C-style strings on device
                  char** h_leftStrings = new char*[tableSize];
                  char** h_rightStrings = new char*[tableSize];
                  
                  // Allocate memory for each string on device
                  for (int i = 0; i < tableSize; i++) {
                      cudaMalloc(&h_leftStrings[i], leftColData[i].size() + 1);
                      cudaMemcpy(h_leftStrings[i], leftColData[i].c_str(), 
                                leftColData[i].size() + 1, cudaMemcpyHostToDevice);
                  }
                  
                  for (int i = 0; i < tableSize; i++) {
                      cudaMalloc(&h_rightStrings[i], rightColData[i].size() + 1);
                      cudaMemcpy(h_rightStrings[i], rightColData[i].c_str(), 
                                rightColData[i].size() + 1, cudaMemcpyHostToDevice);
                  }
                  
                  // Copy arrays of pointers to device
                  char** d_leftStrings, **d_rightStrings;
                  uint8_t* d_results;
                  
                  cudaMalloc(&d_leftStrings, tableSize * sizeof(char*));
                  cudaMalloc(&d_rightStrings, tableSize * sizeof(char*));
                  cudaMalloc(&d_results, tableSize * sizeof(uint8_t));
                  
                  cudaMemcpy(d_leftStrings, h_leftStrings, tableSize * sizeof(char*), cudaMemcpyHostToDevice);
                  cudaMemcpy(d_rightStrings, h_rightStrings, tableSize * sizeof(char*), cudaMemcpyHostToDevice);
                  
                  
                  // Launch string comparison kernel
          
                  cudaEvent_t start, stop;
                  cudaEventCreate(&start);
                  cudaEventCreate(&stop);
                  cudaEventRecord(start);
          
                  compareStringColumnsOneTable<<<gridDim, blockDim>>>(
                      d_leftStrings, d_rightStrings, tableSize, tableSize, d_results, opType);
                  

                  cudaDeviceSynchronize();
                    
                  cudaEventRecord(stop);
                  cudaEventSynchronize(stop);
                  float elapsedTime;
                  cudaEventElapsedTime(&elapsedTime, start, stop);
                  std::cout << "Elapsed time: " << elapsedTime << " milliseconds" << std::endl;
                  cudaEventDestroy(start);
                  cudaEventDestroy(stop);
                  // Copy results back to host
                  cudaMemcpy(resultVector.data(), d_results, tableSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                  
                  // Free device memory
                  for (int i = 0; i < tableSize; i++) {
                      cudaFree(h_leftStrings[i]);
                  }
                  
                  for (int i = 0; i < tableSize; i++) {
                      cudaFree(h_rightStrings[i]);
                  }
                  
                  cudaFree(d_leftStrings);
                  cudaFree(d_rightStrings);
                  cudaFree(d_results);
                  
                  delete[] h_leftStrings;
                  delete[] h_rightStrings;
              }
          }
          
          
          
              }
    }
    
    return resultVector;
}



// Main function with both string and integer handling
std::vector<uint8_t> GPUManager::processComparisonExpr(
    const Table& leftTable, 
    const Table& rightTable,
    const hsql::Expr* expr) 
{
    if (!expr || expr->type != hsql::kExprOperator) {
        throw std::runtime_error("Expected comparison expression");
    }
    
    int leftSize = leftTable.getSize();
    int rightSize = rightTable.getSize();
    int resultSize = leftSize * rightSize;
    
    std::vector<uint8_t> resultVector(resultSize, 0);
    
    // Column-column comparison
    if (expr->expr->type == hsql::kExprColumnRef && expr->expr2->type == hsql::kExprColumnRef) {
        const char* leftColName = expr->expr->name;
        const char* rightColName = expr->expr2->name;
        
        int leftColIndex = findColumnIndex(leftTable, leftColName, expr->expr->table);
        int rightColIndex = findColumnIndex(rightTable, rightColName, expr->expr2->table);
        
        if (leftColIndex == -1 || rightColIndex == -1) {
            // throw std::runtime_error("Column not found in comparison");
            return  std::vector<uint8_t> (resultSize, 1);
        }
        
        // Determine the operator type
        int opType;
        switch (expr->opType) {
            case hsql::OperatorType::kOpEquals: opType = 0; break;
            case hsql::OperatorType::kOpNotEquals: opType = 1; break;
            case hsql::OperatorType::kOpLess: opType = 2; break;
            case hsql::OperatorType::kOpGreater: opType = 3; break;
            case hsql::OperatorType::kOpLessEq: opType = 4; break;
            case hsql::OperatorType::kOpGreaterEq: opType = 5; break;
            default: throw std::runtime_error("Unsupported operator type");
        }
        
        const auto& leftData = leftTable.getData();
        const auto& rightData = rightTable.getData();
        
        // Check if we're dealing with integer or string columns
        bool isIntegerComparison = false;
        if (!leftData.empty() && !rightData.empty()) {
            // Sample the first row of each table to determine type
            isIntegerComparison = isInteger(leftData[0][leftColIndex]) && 
                                 isInteger(rightData[0][rightColIndex]);
        }
        
        // Set up grid and block dimensions for 2D execution
        dim3 blockDim(16, 16);
        dim3 gridDim(
            (leftSize + blockDim.x - 1) / blockDim.x,
            (rightSize + blockDim.y - 1) / blockDim.y
        );
        
        if (isIntegerComparison) {
            // Handle integer columns
            std::vector<int> leftColData(leftSize);
            std::vector<int> rightColData(rightSize);
            
            // Prepare column data
            for (int i = 0; i < leftSize; i++) {
                leftColData[i] = std::stoi(leftData[i][leftColIndex]);
            }
            
            for (int i = 0; i < rightSize; i++) {
                rightColData[i] = std::stoi(rightData[i][rightColIndex]);
            }
            
            // Allocate device memory
            int *d_leftCol, *d_rightCol;
            uint8_t *d_results;
            
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            cudaMalloc(&d_leftCol, leftSize * sizeof(int));
            cudaMalloc(&d_rightCol, rightSize * sizeof(int));
            cudaMalloc(&d_results, resultSize * sizeof(uint8_t));
            
            // Copy data to device
            cudaMemcpy(d_leftCol, leftColData.data(), leftSize * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rightCol, rightColData.data(), rightSize * sizeof(int), cudaMemcpyHostToDevice);
            

            // Launch integer comparison kernel
            compareIntColumns<<<gridDim, blockDim>>>(
                d_leftCol, d_rightCol, leftSize, rightSize, d_results, opType);
                cudaDeviceSynchronize();


                
            // Copy results back to host
            cudaMemcpy(resultVector.data(), d_results, resultSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);
            std::cout << "Elapsed compareIntColumns time: " << elapsedTime << " milliseconds" << std::endl;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            // Free device memory
            cudaFree(d_leftCol);
            cudaFree(d_rightCol);
            cudaFree(d_results);
        } else {
            // Handle string columns
            std::vector<std::string> leftColData(leftSize);
            std::vector<std::string> rightColData(rightSize);
            
            // Prepare column data
            for (int i = 0; i < leftSize; i++) {
                leftColData[i] = leftData[i][leftColIndex];
            }
            
            for (int i = 0; i < rightSize; i++) {
                rightColData[i] = rightData[i][rightColIndex];
            }
            
            // Create array of C-style strings on device
            char** h_leftStrings = new char*[leftSize];
            char** h_rightStrings = new char*[rightSize];
            
            // Allocate memory for each string on device
            for (int i = 0; i < leftSize; i++) {
                cudaMalloc(&h_leftStrings[i], leftColData[i].size() + 1);
                cudaMemcpy(h_leftStrings[i], leftColData[i].c_str(), 
                          leftColData[i].size() + 1, cudaMemcpyHostToDevice);
            }
            
            for (int i = 0; i < rightSize; i++) {
                cudaMalloc(&h_rightStrings[i], rightColData[i].size() + 1);
                cudaMemcpy(h_rightStrings[i], rightColData[i].c_str(), 
                          rightColData[i].size() + 1, cudaMemcpyHostToDevice);
            }
            
            // Copy arrays of pointers to device
            char** d_leftStrings, **d_rightStrings;
            uint8_t* d_results;
            
            cudaMalloc(&d_leftStrings, leftSize * sizeof(char*));
            cudaMalloc(&d_rightStrings, rightSize * sizeof(char*));
            cudaMalloc(&d_results, resultSize * sizeof(uint8_t));
            
            cudaMemcpy(d_leftStrings, h_leftStrings, leftSize * sizeof(char*), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rightStrings, h_rightStrings, rightSize * sizeof(char*), cudaMemcpyHostToDevice);
            
            
            // Launch string comparison kernel

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            compareStringColumns<<<gridDim, blockDim>>>(
                d_leftStrings, d_rightStrings, leftSize, rightSize, d_results, opType);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);
            std::cout << "Elapsed time: " << elapsedTime << " milliseconds" << std::endl;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            // Copy results back to host
            cudaMemcpy(resultVector.data(), d_results, resultSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            
            // Free device memory
            for (int i = 0; i < leftSize; i++) {
                cudaFree(h_leftStrings[i]);
            }
            
            for (int i = 0; i < rightSize; i++) {
                cudaFree(h_rightStrings[i]);
            }
            
            cudaFree(d_leftStrings);
            cudaFree(d_rightStrings);
            cudaFree(d_results);
            
            delete[] h_leftStrings;
            delete[] h_rightStrings;
        }
    }
    
    return resultVector;
}

std::vector<uint8_t> GPUManager::processBinaryExpr(
    const Table& leftTable, 
    const Table& rightTable,
    const hsql::Expr* expr) 
{
    // This is a simplified implementation
    if (expr->type == hsql::kExprOperator) {
        if (expr->opType == hsql::OperatorType::kOpAnd || expr->opType == hsql::OperatorType::kOpOr) {
              // Process binary kOpAnd/kOpOr operations
              auto leftResults = processBinaryExpr(leftTable, rightTable, expr->expr);
              auto rightResults = processBinaryExpr(leftTable, rightTable, expr->expr2);
              
              int resultSize = leftTable.getSize() * rightTable.getSize();
              
              // Create device vectors
              uint8_t *d_leftResults, *d_rightResults, *d_output;
              cudaMalloc(&d_leftResults, resultSize * sizeof(uint8_t));
              cudaMalloc(&d_rightResults, resultSize * sizeof(uint8_t));
              cudaMalloc(&d_output, resultSize * sizeof(uint8_t));
              
              // Copy data to device
              cudaMemcpy(d_leftResults, leftResults.data(), resultSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
              cudaMemcpy(d_rightResults, rightResults.data(), resultSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
              
              // Set up kernel execution parameters
              int blockSize = 256;
              int numBlocks = (resultSize + blockSize - 1) / blockSize;
              
              // Execute kernel
              uint8_t isAnd = expr->opType == hsql::OperatorType::kOpAnd ? 1 : 0;
              combineResults<<<numBlocks, blockSize>>>(d_leftResults, d_rightResults, d_output, resultSize, isAnd);
              
              // Copy results back to host
              std::vector<uint8_t> resultVector(resultSize);
              cudaMemcpy(resultVector.data(), d_output, resultSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
              
              // Free device memory
              cudaFree(d_leftResults);
              cudaFree(d_rightResults);
              cudaFree(d_output);
              
              return resultVector;
        } else {
            return processComparisonExpr(leftTable, rightTable, expr);
        }
    }
    
    // Default case - no conditions
    int resultSize = leftTable.getSize() * rightTable.getSize();
    return std::vector<uint8_t>(resultSize, 1);  // 1 means true
}
