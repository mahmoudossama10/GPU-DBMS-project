#include "../../include/QueryProcessing/GPU.hpp"
#include "../../include/Utilities/ErrorHandling.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
// CUDA kernels
const int GPUManager::BATCH_SIZE = 500;

// Helper for string comparison on device
__device__ int strcmp_device(const char* str1, const char* str2) {
    while (*str1 && (*str1 == *str2)) {
        str1++;
        str2++;
    }
    return *(const unsigned char*)str1 - *(const unsigned char*)str2;
}

// Optimized kernel utilizing shared memory and coalesced access
__global__ void evaluateComparisonBatchOptimized(
    const int64_t* __restrict__ leftColumn,
    const int64_t* __restrict__ rightColumn,
    const int* __restrict__ tableSizes,
    int numTables,
    int leftTableIdx,
    int rightTableIdx,
    int opType,
    uint8_t* __restrict__ results,
    int batchSize,
    int leftBatchSize,
    int rightBatchSize)
{
    // Use shared memory for frequently accessed table metadata
    __shared__ int sharedTableSizes[32]; // Assuming max 32 tables
    
    // Load table sizes into shared memory (only first few threads)
    if (threadIdx.x < numTables) {
        sharedTableSizes[threadIdx.x] = tableSizes[threadIdx.x];
    }
    __syncthreads();
    
    // Process multiple elements per thread for better efficiency
    const int elementsPerThread = 4;
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalThreads = gridDim.x * blockDim.x;
    
    // Each thread processes multiple elements
    for (int i = 0; i < elementsPerThread; i++) {
        int idx = threadId * elementsPerThread + i;
        
        if (idx < batchSize) {
            // Calculate indices for each table from the flattened index
            int indices[32]; // Assuming maximum 32 tables
            int remainingIdx = idx;
            
            #pragma unroll 8  // Unroll for common case (up to 8 tables)
            for (int t = 0; t < numTables; t++) {
                int tableSize = sharedTableSizes[t];
                indices[t] = remainingIdx % tableSize;
                remainingIdx /= tableSize;
            }

            // Get the actual values to compare from the specific tables
            int64_t leftValue = leftColumn[indices[leftTableIdx]];
            int64_t rightValue = rightColumn[indices[rightTableIdx]];
            
            // Evaluate the comparison (branch-free implementation for less divergence)
            uint8_t match;
            
            switch (opType) {
                case 0: // Equals
                    match = (leftValue == rightValue);
                    break;
                case 1: // Not Equals
                    match = (leftValue != rightValue);
                    break;
                case 2: // Less Than
                    match = (leftValue < rightValue);
                    break;
                case 3: // Greater Than
                    match = (leftValue > rightValue);
                    break;
                case 4: // Less Than or Equals
                    match = (leftValue <= rightValue);
                    break;
                case 5: // Greater Than or Equals
                    match = (leftValue >= rightValue);
                    break;
                default:
                    match = 0;
            }
            
            // Store the result
            results[idx] = match;
        }
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

GPUManager::~GPUManager() {
    // Clean up any GPU resources if needed
}

bool GPUManager::isGPUAvailable() const {
    return hasGPU_;
}

int GPUManager::findColumnIndex(const Table& table, const char* columnName, const char* tableName) {
    const auto& headers = table.getHeaders();
    
    for (int i = 0; i < headers.size(); i++) {
        // If table name is specified, check for "tableName.columnName" format
        if (tableName) {
            std::string fullColumnName = std::string(tableName) + "." + std::string(columnName);
            if (headers[i] == fullColumnName || 
                (headers[i] == std::string(columnName) && table.getAlias() == tableName) || 
                (headers[i] == std::string(columnName) && table.getName() == tableName)) {
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

std::shared_ptr<Table> GPUManager::executeBatchedJoin(
    const std::vector<std::shared_ptr<Table>>& tables,
    const hsql::Expr* joinConditions) {
    
    if (!hasGPU_) {
        throw std::runtime_error("GPU operations not available");
    }
    
    std::vector<std::vector<unionV>> resultData;
    
    // Calculate total number of combinations
    int totalCombinations = 1;
    std::vector<int> tableSizes;
    for (const auto& table : tables) {
        tableSizes.push_back(table->getSize());
        totalCombinations *= table->getSize();
    }
    
    // Initialize batch indices for each table
    std::vector<std::vector<int>> batchIndices(tables.size());
    
    // Process in batches
    for (int t0 = 0; t0 < tableSizes[0]; t0 += BATCH_SIZE) {
        int batchSize0 = std::min(BATCH_SIZE, tableSizes[0] - t0);
        batchIndices[0].clear();
        
        for (int i = 0; i < batchSize0; i++) {
            batchIndices[0].push_back(t0 + i);
        }
        
        // Process batches for table 1 and beyond recursively
        processBatchesRecursive(tables, batchIndices, joinConditions, resultData, 1);
    }
    
    // Create result table with appropriate column types
    std::unordered_map<std::string, std::vector<unionV>> columnData;
    std::unordered_map<std::string, ColumnType> columnTypes;
    
    auto headers = combineMultipleHeaders(tables);
    
    // Initialize column data structure
    for (size_t col = 0; col < headers.size(); col++) {
        columnData[headers[col]] = std::vector<unionV>(resultData.size());
    }
    
    // Populate column data
    for (size_t row = 0; row < resultData.size(); row++) {
        for (size_t col = 0; col < headers.size(); col++) {
            columnData[headers[col]][row] = resultData[row][col];
        }
    }
    
    // Determine column types by looking at source tables
    int colOffset = 0;
    for (const auto& table : tables) {
        const auto& tableHeaders = table->getHeaders();
        const auto& tableTypes = table->getColumnTypes();
        
        for (const auto& header : tableHeaders) {
            std::string resultHeader = headers[colOffset];
            auto it = tableTypes.find(header);
            if (it != tableTypes.end()) {
                columnTypes[resultHeader] = it->second;
            } else {
                // Default to string if type not known
                columnTypes[resultHeader] = ColumnType::STRING;
            }
            colOffset++;
        }
    }
    
    return std::make_shared<Table>("joined_result", headers, columnData, columnTypes);
}

// Helper method to recursively process batches across multiple tables
void GPUManager::processBatchesRecursive(
    const std::vector<std::shared_ptr<Table>>& tables,
    std::vector<std::vector<int>>& batchIndices,
    const hsql::Expr* joinConditions,
    std::vector<std::vector<unionV>>& resultData,
    int tableIndex) {
    
    // Base case: processed all tables in this batch combination
    if (tableIndex >= tables.size()) {
        // Process this specific batch combination
        auto startProcess = std::chrono::high_resolution_clock::now();

        auto batchResults = processBatch(tables, batchIndices, joinConditions);


        auto endProcess = std::chrono::high_resolution_clock::now();
std::chrono::duration<double, std::milli> processTime = endProcess - startProcess;
std::cout << "processBatch time: " << processTime.count() << " ms" << std::endl;

auto startFilter = std::chrono::high_resolution_clock::now();
        // Extract matching rows from the batch
        std::vector<std::vector<int>> selectedCombinations;
        int totalBatchSize = 1;
        for (const auto& indices : batchIndices) {
            totalBatchSize *= indices.size();
        }

        // Find all combinations that matched (where batchResults[i] == 1)
        for (int i = 0; i < totalBatchSize; i++) {
            if (batchResults[i] == 1) {
                std::vector<int> combination;
                int index = i;
                for (int t = batchIndices.size() - 1; t >= 0; t--) {
                    int tableSize = batchIndices[t].size();
                    combination.insert(combination.begin(), batchIndices[t][index % tableSize]);
                    index /= tableSize;
                }
                selectedCombinations.push_back(combination);
            }
        }


        auto endFilter = std::chrono::high_resolution_clock::now();
std::chrono::duration<double, std::milli> filterTime = endFilter - startFilter;
std::cout << "Filtering matches time: " << filterTime.count() << " ms" << std::endl;
        // Merge selected rows into result

        auto startMerge = std::chrono::high_resolution_clock::now();

        auto batchData = mergeBatchResults(tables, selectedCombinations);
        resultData.insert(resultData.end(), batchData.begin(), batchData.end());

        auto endMerge = std::chrono::high_resolution_clock::now();
std::chrono::duration<double, std::milli> mergeTime = endMerge - startMerge;
std::cout << "Merging results time: " << mergeTime.count() << " ms" << std::endl;

        return;
    }
    
    // Recursive case: process next table in batches
    int tableSize = tables[tableIndex]->getSize();
    for (int t = 0; t < tableSize; t += BATCH_SIZE) {
        int batchSize = std::min(BATCH_SIZE, tableSize - t);
        
        // Set up indices for this batch of the current table
        batchIndices[tableIndex].clear();
        for (int i = 0; i < batchSize; i++) {
            batchIndices[tableIndex].push_back(t + i);
        }
        
        // Process next table
        processBatchesRecursive(tables, batchIndices, joinConditions, resultData, tableIndex + 1);
    }
}

// Process a specific batch combination across all tables
std::vector<uint8_t> GPUManager::processBatch(
    const std::vector<std::shared_ptr<Table>>& tables,
    const std::vector<std::vector<int>>& batchIndices,
    const hsql::Expr* conditions) {
    
    // Calculate batch size
    int batchSize = 1;
    for (const auto& indices : batchIndices) {
        batchSize *= indices.size();
    }
    
    // Initialize all results to 1 (true)
    std::vector<uint8_t> results(batchSize, 1);
    
    // Base case: no conditions
    if (!conditions) {
        return results;
    }
    
    // Process conditions based on operation type
    if (conditions->type == hsql::kExprOperator) {
        if (conditions->opType == hsql::OperatorType::kOpAnd || 
            conditions->opType == hsql::OperatorType::kOpOr) {
            // Combine binary operations (AND/OR)
            auto leftResults = processBatch(tables, batchIndices, conditions->expr);
            auto rightResults = processBatch(tables, batchIndices, conditions->expr2);
            
            // Combine results on GPU
            uint8_t *d_leftResults, *d_rightResults, *d_output;
            cudaMalloc(&d_leftResults, batchSize * sizeof(uint8_t));
            cudaMalloc(&d_rightResults, batchSize * sizeof(uint8_t));
            cudaMalloc(&d_output, batchSize * sizeof(uint8_t));
            
            cudaMemcpy(d_leftResults, leftResults.data(), batchSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rightResults, rightResults.data(), batchSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
            
            int blockSize = 256;
            int numBlocks = (batchSize + blockSize - 1) / blockSize;
            
            uint8_t isAnd = conditions->opType == hsql::OperatorType::kOpAnd ? 1 : 0;
            combineResults<<<numBlocks, blockSize>>>(d_leftResults, d_rightResults, d_output, batchSize, isAnd);
            
            cudaMemcpy(results.data(), d_output, batchSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            
            cudaFree(d_leftResults);
            cudaFree(d_rightResults);
            cudaFree(d_output);
        } else {
            // Process simple comparison between columns
            results = evaluateConditionOnBatch(tables, batchIndices, conditions);
        }
    }
    
    return results;
}

// Improved version of evaluateConditionOnBatch for column-major data and integer columns
std::vector<uint8_t> GPUManager::evaluateConditionOnBatch(
    const std::vector<std::shared_ptr<Table>>& tables,
    const std::vector<std::vector<int>>& batchIndices,
    const hsql::Expr* condition) {
    
    // Calculate total batch size (cartesian product of all tables)
    int batchSize = 1;
    for (const auto& indices : batchIndices) {
        batchSize *= indices.size();
    }
    
    // Default to all true
    std::vector<uint8_t> results(batchSize, 1);
    
    // Handle comparison operators
    if (condition->type == hsql::kExprOperator && 
        condition->expr->type == hsql::kExprColumnRef && 
        condition->expr2->type == hsql::kExprColumnRef) {
        
        // Find which tables and columns are involved
        int leftTableIdx = -1, rightTableIdx = -1;
        std::string leftColName, rightColName;
        
        // Identify tables and columns involved in this condition
        for (int i = 0; i < tables.size(); i++) {
            int col = findColumnIndex(*tables[i], condition->expr->name, condition->expr->table);
            if (col != -1) {
                leftTableIdx = i;
                leftColName = tables[i]->getHeaders()[col];
            }
            
            col = findColumnIndex(*tables[i], condition->expr2->name, condition->expr2->table);
            if (col != -1) {
                rightTableIdx = i;
                rightColName = tables[i]->getHeaders()[col];
            }
        }
        
        // If we couldn't find the columns, return all true
        if (leftTableIdx == -1 || rightTableIdx == -1 || leftColName.empty() || rightColName.empty()) {
            return results;
        }
        
        // Get operator type
        int opType;
        switch (condition->opType) {
            case hsql::OperatorType::kOpEquals: opType = 0; break;
            case hsql::OperatorType::kOpNotEquals: opType = 1; break;
            case hsql::OperatorType::kOpLess: opType = 2; break;
            case hsql::OperatorType::kOpGreater: opType = 3; break;
            case hsql::OperatorType::kOpLessEq: opType = 4; break;
            case hsql::OperatorType::kOpGreaterEq: opType = 5; break;
            default: throw std::runtime_error("Unsupported operator type");
        }
        
        // Check if columns are integers - we're only handling integers in this implementation
        const auto& leftTable = tables[leftTableIdx];
        const auto& rightTable = tables[rightTableIdx];
        
        // Verify column types are integers
        if (leftTable->getColumnType(leftColName) != ColumnType::INTEGER || 
            rightTable->getColumnType(rightColName) != ColumnType::INTEGER) {
            // We're only implementing integer comparisons in this version
            return results;
        }
        
        // Extract the batch data for the columns we need to compare
        int leftBatchSize = batchIndices[leftTableIdx].size();
        int rightBatchSize = batchIndices[rightTableIdx].size();
        
        // Get integer column data directly from the Table class
        // (using the cached int columns from the Table implementation)
        const auto& leftColumnData = leftTable->getIntColumn(leftColName);
        const auto& rightColumnData = rightTable->getIntColumn(rightColName);
        
        // No need to convert strings to integers, as we now have direct access to int data
        
        // Prepare data for GPU - allocate host arrays for the batch values
        std::vector<int64_t> leftBatchData(leftBatchSize);
        std::vector<int64_t> rightBatchData(rightBatchSize);
        
        // Extract the batch values from the full columns using the batch indices
        for (int i = 0; i < leftBatchSize; i++) {
            int rowIdx = batchIndices[leftTableIdx][i];
            leftBatchData[i] = leftTable->getInteger(leftColName, rowIdx);
        }
        
        for (int i = 0; i < rightBatchSize; i++) {
            int rowIdx = batchIndices[rightTableIdx][i];
            rightBatchData[i] = rightTable->getInteger(rightColName, rowIdx);
        }
        
        // auto leftBatchData = leftTable->getData().at(leftColName);
        // auto rightBatchData = rightTable->getData().at(rightColName);

        // Stream for asynchronous operations
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // Allocate device memory
        int64_t *d_leftCol, *d_rightCol;
        uint8_t *d_results;
        int* d_tableSizes;
        
        cudaMalloc(&d_leftCol, leftBatchSize * sizeof(int64_t));
        cudaMalloc(&d_rightCol, rightBatchSize * sizeof(int64_t));
        cudaMalloc(&d_results, batchSize * sizeof(uint8_t));
        
        // Initialize results to 1 (true)
        cudaMemsetAsync(d_results, 1, batchSize * sizeof(uint8_t), stream);
        
        // Create table indices information
        std::vector<int> tableSizes;
        for (const auto& indices : batchIndices) {
            tableSizes.push_back(indices.size());
        }
        
        cudaMalloc(&d_tableSizes, tableSizes.size() * sizeof(int));
        
        // Async memory transfers
        cudaMemcpyAsync(d_leftCol, leftBatchData.data(), leftBatchSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_rightCol, rightBatchData.data(), rightBatchSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_tableSizes, tableSizes.data(), tableSizes.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
        
        // Calculate threads needed
        const int blockSize = 256;
        
        // Calculate grid size - we process multiple elements per thread
        const int elementsPerThread = 4;
        const int effectiveThreads = (batchSize + elementsPerThread - 1) / elementsPerThread;
        const int gridSize = (effectiveThreads + blockSize - 1) / blockSize;
        
        // Launch optimized kernel
        evaluateComparisonBatchOptimized<<<gridSize, blockSize, 0, stream>>>(
            d_leftCol, d_rightCol, 
            d_tableSizes, tables.size(),
            leftTableIdx, rightTableIdx,
            opType, d_results, batchSize,
            leftBatchSize, rightBatchSize);
        
        // Copy results back to host asynchronously
        cudaMemcpyAsync(results.data(), d_results, batchSize * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
        
        // Synchronize to ensure results are ready
        cudaStreamSynchronize(stream);
        
        // Free resources
        cudaFree(d_leftCol);
        cudaFree(d_rightCol);
        cudaFree(d_results);
        cudaFree(d_tableSizes);
        cudaStreamDestroy(stream);
    }
    
    return results;
}

// Method to combine headers from multiple tables
std::vector<std::string> GPUManager::combineMultipleHeaders(
    const std::vector<std::shared_ptr<Table>>& tables) {
    
    std::vector<std::string> headers;
    
    for (const auto& table : tables) {
        const auto& tableHeaders = table->getHeaders();
        const std::string& alias = table->getAlias();
        
        for (const auto& header : tableHeaders) {
            headers.push_back(alias.empty() ? header : alias + "." + header);
        }
    }
    
    return headers;
}

// Method to merge rows from tables based on selected indices - adapted for unionV
std::vector<std::vector<unionV>> GPUManager::mergeBatchResults(
    const std::vector<std::shared_ptr<Table>>& tables,
    const std::vector<std::vector<int>>& selectedIndices) {
    
    std::vector<std::vector<unionV>> results;
    results.reserve(selectedIndices.size());

    // Precompute total columns per combination
    int totalCols = 0;
    for (const auto& table : tables) {
        totalCols += table->getHeaders().size();
    }

    for (const auto& combination : selectedIndices) {
        std::vector<unionV> mergedRow;
        mergedRow.reserve(totalCols);

        for (int t = 0; t < tables.size(); t++) {
            const auto& row = tables[t]->getRow(combination[t]);
            mergedRow.insert(mergedRow.end(), row.begin(), row.end());
        }

        results.push_back(std::move(mergedRow));
    }

    return results;
}



// // Updated kernel to handle full batch size
// __global__ void evaluateComparisonBatch(
//     const int* leftColumn,
//     const int* rightColumn,
//     const int* tableSizes,
//     int numTables,
//     int leftTableIdx,
//     int rightTableIdx,
//     int opType,
//     uint8_t* results,
//     int batchSize)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (idx < batchSize) {
//         // Calculate indices for each table from the flattened index
//         int indices[32]; // Assuming maximum 32 tables
//         int remainingIdx = idx;
        
//         // Convert linear index to multi-dimensional indices
//         for (int t = 0; t < numTables; t++) {
//             int tableSize = tableSizes[t];
//             indices[t] = remainingIdx % tableSize;
//             remainingIdx /= tableSize;
//         }

//         // Get the actual values to compare from the specific tables
//         int leftValue = leftColumn[indices[leftTableIdx]];
//         int rightValue = rightColumn[indices[rightTableIdx]];
        
//         // Evaluate the comparison based on operation type
//         uint8_t match = 0;
//         switch (opType) {
//             case 0: // Equals
//                 match = (leftValue == rightValue) ? 1 : 0;
//                 break;
//             case 1: // Not Equals
//                 match = (leftValue != rightValue) ? 1 : 0;
//                 break;
//             case 2: // Less Than
//                 match = (leftValue < rightValue) ? 1 : 0;
//                 break;
//             case 3: // Greater Than
//                 match = (leftValue > rightValue) ? 1 : 0;
//                 break;
//             case 4: // Less Than or Equals
//                 match = (leftValue <= rightValue) ? 1 : 0;
//                 break;
//             case 5: // Greater Than or Equals
//                 match = (leftValue >= rightValue) ? 1 : 0;
//                 break;
//         }

//         // // Print the match result
//         // printf("Thread %d: match = %d\n", idx, match);
        
//         // Store the result
//         results[idx] = match;
//     }
// }


// __global__ void compareStringColumns(
//      char** leftColumn, 
//      char** rightColumn,
//     int leftSize, 
//     int rightSize,
//     uint8_t* results, 
//     int opType) 
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
    
//     if (i < leftSize && j < rightSize) {
//         uint8_t match = 0;
        
//         switch (opType) {
//             case 0: // Equals
//                 match = (strcmp_device(leftColumn[i], rightColumn[j]) == 0) ? 1 : 0;
//                 break;
//             case 1: // Not Equals
//                 match = (strcmp_device(leftColumn[i], rightColumn[j]) != 0) ? 1 : 0;
//                 break;
//             case 2: // Less Than
//                 match = (strcmp_device(leftColumn[i], rightColumn[j]) < 0) ? 1 : 0;
//                 break;
//             case 3: // Greater Than
//                 match = (strcmp_device(leftColumn[i], rightColumn[j]) > 0) ? 1 : 0;
//                 break;
//             case 4: // Less Than or Equals
//                 match = (strcmp_device(leftColumn[i], rightColumn[j]) <= 0) ? 1 : 0;
//                 break;
//             case 5: // Greater Than or Equals
//                 match = (strcmp_device(leftColumn[i], rightColumn[j]) >= 0) ? 1 : 0;
//                 break;
//         }
        
//         results[i * rightSize + j] = match;
//     }
// }



// __global__ void compareStringColumnsOneTable(
//     char** leftColumn, 
//     char** rightColumn,
//    int leftSize, 
//    int rightSize,
//    uint8_t* results, 
//    int opType) 
// {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
   
//    if (i < leftSize) {
//        uint8_t match = 0;
       
//        switch (opType) {
//            case 0: // Equals
//                match = (strcmp_device(leftColumn[i], rightColumn[i]) == 0) ? 1 : 0;
//                break;
//            case 1: // Not Equals
//                match = (strcmp_device(leftColumn[i], rightColumn[i]) != 0) ? 1 : 0;
//                break;
//            case 2: // Less Than
//                match = (strcmp_device(leftColumn[i], rightColumn[i]) < 0) ? 1 : 0;
//                break;
//            case 3: // Greater Than
//                match = (strcmp_device(leftColumn[i], rightColumn[i]) > 0) ? 1 : 0;
//                break;
//            case 4: // Less Than or Equals
//                match = (strcmp_device(leftColumn[i], rightColumn[i]) <= 0) ? 1 : 0;
//                break;
//            case 5: // Greater Than or Equals
//                match = (strcmp_device(leftColumn[i], rightColumn[i]) >= 0) ? 1 : 0;
//                break;
//        }
       
//        results[i] = match;
//    }
// }

// __global__ void compareIntColumns(
//     const int* leftColumn, 
//     const int* rightColumn,
//     int leftSize, 
//     int rightSize,
//     uint8_t* results, 
//     int opType,
//     const int* tableSizes,  // Array containing sizes of all tables
//     int numTables,          // Total number of tables
//     int leftTableIdx,       // Index of left table in the tables array
//     int rightTableIdx)      // Index of right table in the tables array
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
    
//     if (i < leftSize && j < rightSize) {
//         uint8_t match = 0;
        
//         // Compare the values based on the operation type
//         switch (opType) {
//             case 0: // Equals
//                 match = (leftColumn[i] == rightColumn[j]) ? 1 : 0;
//                 break;
//             case 1: // Not Equals
//                 match = (leftColumn[i] != rightColumn[j]) ? 1 : 0;
//                 break;
//             case 2: // Less Than
//                 match = (leftColumn[i] < rightColumn[j]) ? 1 : 0;
//                 break;
//             case 3: // Greater Than
//                 match = (leftColumn[i] > rightColumn[j]) ? 1 : 0;
//                 break;
//             case 4: // Less Than or Equals
//                 match = (leftColumn[i] <= rightColumn[j]) ? 1 : 0;
//                 break;
//             case 5: // Greater Than or Equals
//                 match = (leftColumn[i] >= rightColumn[j]) ? 1 : 0;
//                 break;
//         }
        
//         // Calculate the output index in the flattened result array for multiple tables
//         // We need to compute the position in the N-dimensional space and flatten it
        
//         // First, initialize the indices for each table dimension
//         int indices[32]; // Assuming maximum 32 tables, adjust as needed
//         for (int t = 0; t < numTables; t++) {
//             indices[t] = 0;
//         }
        
//         // Set the actual indices for the tables involved in the comparison
//         indices[leftTableIdx] = i;
//         indices[rightTableIdx] = j;
        
//         // Calculate the flattened index
//         int flatIndex = 0;
//         int stride = 1;
        
//         // Calculate the flattened index using row-major ordering
//         for (int t = numTables - 1; t >= 0; t--) {
//             flatIndex += indices[t] * stride;
//             stride *= tableSizes[t];
//         }
        
//         // Store the result at the calculated position
//         results[flatIndex] = match;
//     }
// }

// __global__ void compareIntColumnsOmeTable(
//     int* leftColumn, 
//     int* rightColumn,
//    int leftSize, 
//    int rightSize,
//    uint8_t* results, 
//    int opType) 
// {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
   
//    if (i < leftSize) {
//        uint8_t match = 0;
       
//        switch (opType) {
//            case 0: // Equals
//                match = (leftColumn[i] == rightColumn[i]) ? 1 : 0;
//                break;
//            case 1: // Not Equals
//                match = (leftColumn[i] != rightColumn[i]) ? 1 : 0;
//                break;
//            case 2: // Less Than
//                match = (leftColumn[i] < rightColumn[i]) ? 1 : 0;
//                break;
//            case 3: // Greater Than
//                match = (leftColumn[i] > rightColumn[i]) ? 1 : 0;
//                break;
//            case 4: // Less Than or Equals
//                match = (leftColumn[i] <= rightColumn[i]) ? 1 : 0;
//                break;
//            case 5: // Greater Than or Equals
//                match = (leftColumn[i] >= rightColumn[i]) ? 1 : 0;
//                break;
//        }
       
//        results[i] = match;
//    }
// }

// __global__ void compareIntWithConstant(
//      int* column, 
//     int constant,
//     int size, 
//     uint8_t* results, 
//     int opType) 
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (i < size) {
//         uint8_t match = 0;
        
//         switch (opType) {
//             case 0: // Equals
//                 match = (column[i] == constant) ? 1 : 0;
//                 break;
//             case 1: // Not Equals
//                 match = (column[i] != constant) ? 1 : 0;
//                 break;
//             case 2: // Less Than
//                 match = (column[i] < constant) ? 1 : 0;
//                 break;
//             case 3: // Greater Than
//                 match = (column[i] > constant) ? 1 : 0;
//                 break;
//             case 4: // Less Than or Equals
//                 match = (column[i] <= constant) ? 1 : 0;
//                 break;
//             case 5: // Greater Than or Equals
//                 match = (column[i] >= constant) ? 1 : 0;
//                 break;
//         }
        
//         results[i] = match;
//     }
// }
// New kernel that uses the improved string storage approach
// __global__ void compareStringWithConstantImproved(
//     const char* stringBuffer,
//     const int* stringOffsets,
//     const char* constant,
//     int size,
//     uint8_t* results,
//     int opType)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (i < size) {
//         // Get pointer to the current string using the offset
//         const char* currentString = stringBuffer + stringOffsets[i];
        
//         int comparison = 0;
//         // Compare strings
//         while (*currentString == *constant && *currentString && *constant) {
//             currentString++;
//             constant++;
//         }
        
//         comparison = (unsigned char)*currentString - (unsigned char)*constant;
        
//         uint8_t match = 0;
//         switch (opType) {
//             case 0: // Equals
//                 match = (comparison == 0) ? 1 : 0;
//                 break;
//             case 1: // Not Equals
//                 match = (comparison != 0) ? 1 : 0;
//                 break;
//             case 2: // Less Than
//                 match = (comparison < 0) ? 1 : 0;
//                 break;
//             case 3: // Greater Than
//                 match = (comparison > 0) ? 1 : 0;
//                 break;
//             case 4: // Less Than or Equals
//                 match = (comparison <= 0) ? 1 : 0;
//                 break;
//             case 5: // Greater Than or Equals
//                 match = (comparison >= 0) ? 1 : 0;
//                 break;
//         }
        
//         results[i] = match;
//     }
// }




// std::vector<uint8_t> GPUManager::gpuJoinTables(
//     const Table& leftTable, 
//     const Table& rightTable,
//     const hsql::Expr* conditions) 
// {
//     if (!hasGPU_) {
//         throw std::runtime_error("GPU operations not available");
//     }
    
//     int leftSize = leftTable.getSize();
//     int rightSize = rightTable.getSize();
//     int resultSize = leftSize * rightSize;
    
//     std::vector<uint8_t> resultVector(resultSize, 0);
    
//     // Process each condition and combine results
//     if (conditions->type == hsql::kExprOperator) {
//         if (conditions->opType == hsql::OperatorType::kOpAnd || conditions->opType == hsql::OperatorType::kOpOr) {
//             // Process binary kOpAnd/kOpOr operations
//             auto leftResults = processBinaryExpr(leftTable, rightTable, conditions->expr);
//             auto rightResults = processBinaryExpr(leftTable, rightTable, conditions->expr2);
            
//             // Create device vectors
//             uint8_t *d_leftResults, *d_rightResults, *d_output;
//             cudaMalloc(&d_leftResults, resultSize * sizeof(uint8_t));
//             cudaMalloc(&d_rightResults, resultSize * sizeof(uint8_t));
//             cudaMalloc(&d_output, resultSize * sizeof(uint8_t));
            
//             // Copy data to device
//             cudaMemcpy(d_leftResults, leftResults.data(), resultSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
//             cudaMemcpy(d_rightResults, rightResults.data(), resultSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
            
//             // Set up kernel execution parameters
//             int blockSize = 256;
//             int numBlocks = (resultSize + blockSize - 1) / blockSize;
            
//             // Execute kernel
//             uint8_t isAnd = conditions->opType == hsql::OperatorType::kOpAnd ? 1 : 0;
//             combineResults<<<numBlocks, blockSize>>>(d_leftResults, d_rightResults, d_output, resultSize, isAnd);
            
//             // Copy results back to host
//             cudaMemcpy(resultVector.data(), d_output, resultSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            
//             // Free device memory
//             cudaFree(d_leftResults);
//             cudaFree(d_rightResults);
//             cudaFree(d_output);
//         } 
//         else {
//             // Process comparison operation
//             resultVector = processComparisonExpr(leftTable, rightTable, conditions);
//         }
//     }
    
//     return resultVector;
// }

// std::shared_ptr<Table> GPUManager::executeJoin(std::shared_ptr<Table> leftTable,
//                                              std::shared_ptr<Table> rightTable,
//                                              const hsql::Expr* condition) {
//     // Get GPU join mask
//     auto mask = gpuJoinTables(*leftTable, *rightTable, condition);
    
//     // Create result table structure
//     auto headers = combineHeaders(*leftTable, *rightTable);
//     auto data = mergeJoinResults(*leftTable, *rightTable, mask);
    
//     return std::make_shared<Table>(
//         leftTable->getName() + "_joined_" + rightTable->getName(),
//         headers,
//         data
//     );
// }

// std::shared_ptr<Table> GPUManager::applyFilter(const Table& table, 
//                                              const std::vector<uint8_t>& mask) 
// {
//     return std::make_shared<Table>(
//         table.getName() + "_filtered",
//         table.getHeaders(),
//         mergeFilterResults(table, mask)
//     );
// }

// std::vector<std::vector<std::string>> GPUManager::mergeFilterResults(
//     const Table& table,
//     const std::vector<uint8_t>& mask) const 
// {
//     std::vector<std::vector<std::string>> result;
    
//     #pragma omp parallel for
//     for (int i = 0; i < mask.size(); ++i) {
//         if (mask[i]) {
//             #pragma omp critical
//             result.push_back(table.getRow(i));
//         }
//     }
    
//     return result;
// }

// std::vector<std::string> GPUManager::combineHeaders(const Table& left,
//                                                   const Table& right) const {
//     std::vector<std::string> headers;
    
//     // Add left headers with alias
//     for (const auto& h : left.getHeaders()) {
//         headers.push_back(left.getAlias().empty() ? h : left.getAlias() + "." + h);
//     }
    
//     // Add right headers with alias
//     for (const auto& h : right.getHeaders()) {
//         headers.push_back(right.getAlias().empty() ? h : right.getAlias() + "." + h);
//     }
    
//     return headers;
// }

// std::vector<std::vector<std::string>> GPUManager::mergeJoinResults(
//     const Table& left,
//     const Table& right,
//     const std::vector<uint8_t>& mask) const {
    
//     std::vector<std::vector<std::string>> result;
//     const int rightSize = right.getSize();
    
//     #pragma omp parallel for
//     for (int idx = 0; idx < mask.size(); ++idx) {
//         if (mask[idx]) {
//             // Calculate row indices
//             const int leftIdx = idx / rightSize;
//             const int rightIdx = idx % rightSize;
            
//             // Combine rows
//             auto combined = left.getRow(leftIdx);
//             const auto& rightRow = right.getRow(rightIdx);
//             combined.insert(combined.end(), rightRow.begin(), rightRow.end());
            
//             #pragma omp critical
//             result.push_back(std::move(combined));
//         }
//     }
    
//     return result;
// }

// std::vector<uint8_t> GPUManager::gpuFilterTable(
//     const Table& table,
//     const hsql::Expr* conditions) 
// {
//     if (!hasGPU_) {
//         throw std::runtime_error("GPU operations not available");
//     }

    
//     int tableSize = table.getSize();
//     std::vector<uint8_t> resultVector(tableSize, 0);
    
//     // Process the conditions (simplified for the example)
//     if (conditions->type == hsql::kExprOperator) {
//         // Handle comparison with constant
//         if (conditions->expr->type == hsql::kExprColumnRef && 
//             (conditions->expr2->type == hsql::kExprLiteralInt || 
//              conditions->expr2->type == hsql::kExprLiteralString)) {
            
//             const char* columnName = conditions->expr->name;
//             int columnIndex = findColumnIndex(table, columnName, conditions->expr->table);
            
//             if (columnIndex == -1) {
//                 throw std::runtime_error("Column not found: " + std::string(columnName));
//             }
            
//             uint8_t* d_results;
//             cudaMalloc(&d_results, tableSize * sizeof(uint8_t));
//             cudaMemset(d_results, 0, tableSize * sizeof(uint8_t));
            
//             int blockSize = 256;
//             int numBlocks = (tableSize + blockSize - 1) / blockSize;
            
//             // Convert operator type to our internal representation
//             int opType;
//             switch (conditions->opType) {
//                 case hsql::OperatorType::kOpEquals: opType = 0; break;
//                 case hsql::OperatorType::kOpNotEquals: opType = 1; break;
//                 case hsql::OperatorType::kOpLess: opType = 2; break;
//                 case hsql::OperatorType::kOpGreater: opType = 3; break;
//                 case hsql::OperatorType::kOpLessEq: opType = 4; break;
//                 case hsql::OperatorType::kOpGreaterEq: opType = 5; break;
//                 default: throw std::runtime_error("Unsupported operator type");
//             }
            
//             // For integer comparison
//             if (conditions->expr2->type == hsql::kExprLiteralInt) {
//                 int constant = conditions->expr2->ival;
                
//                 // Prepare column data for GPU
//                 std::vector<int> columnData(tableSize);
//                 const auto& data = table.getData();
//                 for (int i = 0; i < tableSize; i++) {
//                     columnData[i] = std::stoi(data[i][columnIndex]);
//                 }
                
//                 int* d_column;
//                 cudaMalloc(&d_column, tableSize * sizeof(int));
//                 cudaMemcpy(d_column, columnData.data(), tableSize * sizeof(int), cudaMemcpyHostToDevice);
                
//                 compareIntWithConstant<<<numBlocks, blockSize>>>(d_column, constant, tableSize, d_results, opType);
//                 cudaFree(d_column);
//             }
//             // For string comparison
//             else if (conditions->expr2->type == hsql::kExprLiteralString) {
//                 const char* constant = conditions->expr2->name;
//                 int constantLen = strlen(constant) + 1; // +1 for null terminator
                
//                 // Get string data from the table
//                 const auto& data = table.getData();
                
//                 // Calculate total buffer size needed for all strings
//                 int totalBufferSize = 0;
//                 std::vector<int> stringLengths(tableSize);
//                 std::vector<int> stringOffsets(tableSize);
                
//                 for (int i = 0; i < tableSize; i++) {
//                     stringLengths[i] = data[i][columnIndex].length() + 1; // +1 for null terminator
//                     stringOffsets[i] = totalBufferSize;
//                     totalBufferSize += stringLengths[i];
//                 }
                
//                 // Create a buffer for all strings
//                 char* stringBuffer = new char[totalBufferSize];
                
//                 // Copy strings to the buffer
//                 for (int i = 0; i < tableSize; i++) {
//                     strcpy(stringBuffer + stringOffsets[i], data[i][columnIndex].c_str());
//                 }
                
//                 // Allocate device memory
//                 char* d_stringBuffer;
//                 int* d_stringOffsets;
//                 char* d_constant;
                
//                 cudaMalloc(&d_stringBuffer, totalBufferSize);
//                 cudaMalloc(&d_stringOffsets, tableSize * sizeof(int));
//                 cudaMalloc(&d_constant, constantLen);
                
//                 // Copy data to device
//                 cudaMemcpy(d_stringBuffer, stringBuffer, totalBufferSize, cudaMemcpyHostToDevice);
//                 cudaMemcpy(d_stringOffsets, stringOffsets.data(), tableSize * sizeof(int), cudaMemcpyHostToDevice);
//                 cudaMemcpy(d_constant, constant, constantLen, cudaMemcpyHostToDevice);
                
//                 // Launch kernel with improved string handling
//                 compareStringWithConstantImproved<<<numBlocks, blockSize>>>(
//                     d_stringBuffer, d_stringOffsets, d_constant, tableSize, d_results, opType);
                
//                 cudaError_t err = cudaGetLastError();
//                 if (err != cudaSuccess) {
//                     std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
//                 }
                
//                 // Free allocated memory
//                 delete[] stringBuffer;
//                 cudaFree(d_stringBuffer);
//                 cudaFree(d_stringOffsets);
//                 cudaFree(d_constant);
//             }
            
//             // Copy results back
//             cudaMemcpy(resultVector.data(), d_results, tableSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
//             cudaFree(d_results);
//         }


//         else{
//             // Column-column comparison
//             if (conditions->expr->type == hsql::kExprColumnRef && conditions->expr2->type == hsql::kExprColumnRef) {
//               const char* leftColName = conditions->expr->name;
//               const char* rightColName = conditions->expr2->name;
              
//               int leftColIndex = findColumnIndex(table, leftColName, conditions->expr->table);
//               int rightColIndex = findColumnIndex(table, rightColName, conditions->expr2->table);
              
              
//               // Determine the operator type
//               int opType;
//               switch (conditions->opType) {
//                   case hsql::OperatorType::kOpEquals: opType = 0; break;
//                   case hsql::OperatorType::kOpNotEquals: opType = 1; break;
//                   case hsql::OperatorType::kOpLess: opType = 2; break;
//                   case hsql::OperatorType::kOpGreater: opType = 3; break;
//                   case hsql::OperatorType::kOpLessEq: opType = 4; break;
//                   case hsql::OperatorType::kOpGreaterEq: opType = 5; break;
//                   default: throw std::runtime_error("Unsupported operator type");
//               }
              
//               const auto& leftData = table.getData();
//               const auto& rightData = table.getData();
              
//               // Check if we're dealing with integer or string columns
//               bool isIntegerComparison = false;
//               if (!leftData.empty() && !rightData.empty()) {
//                   // Sample the first row of each table to determine type
//                   isIntegerComparison = isInteger(leftData[0][leftColIndex]) && isInteger(rightData[0][rightColIndex]);
//               }
              
//               // Set up grid and block dimensions for 2D execution
//               dim3 blockDim(16, 16);
//               dim3 gridDim(
//                   (tableSize + blockDim.x - 1) / blockDim.x,
//                   (tableSize + blockDim.y - 1) / blockDim.y
//               );
              
//               if (isIntegerComparison) {
//                   // Handle integer columns
//                   std::vector<int> leftColData(tableSize);
//                   std::vector<int> rightColData(tableSize);
                  
//                   // Prepare column data
//                   for (int i = 0; i < tableSize; i++) {
//                       leftColData[i] = std::stoi(leftData[i][leftColIndex]);
//                   }
                  
//                   for (int i = 0; i < tableSize; i++) {
//                       rightColData[i] = std::stoi(rightData[i][rightColIndex]);
//                   }
                  
//                   // Allocate device memory
//                   int *d_leftCol, *d_rightCol;
//                   uint8_t *d_results;
                  
//                   cudaMalloc(&d_leftCol, tableSize * sizeof(int));
//                   cudaMalloc(&d_rightCol, tableSize * sizeof(int));
//                   cudaMalloc(&d_results, tableSize * sizeof(uint8_t));
                  
//                   // Copy data to device
//                   cudaMemcpy(d_leftCol, leftColData.data(), tableSize * sizeof(int), cudaMemcpyHostToDevice);
//                   cudaMemcpy(d_rightCol, rightColData.data(), tableSize * sizeof(int), cudaMemcpyHostToDevice);
                  
//                   cudaEvent_t start, stop;
//                   cudaEventCreate(&start);
//                   cudaEventCreate(&stop);
//                   cudaEventRecord(start);
//                   // Launch integer comparison kernel
//                   compareIntColumnsOmeTable<<<gridDim, blockDim>>>(
//                       d_leftCol, d_rightCol, tableSize, tableSize, d_results, opType);
                  
//                       cudaEventRecord(stop);
//                       cudaEventSynchronize(stop);
//                       float elapsedTime;
//                       cudaEventElapsedTime(&elapsedTime, start, stop);
//                       std::cout << "Elapsed compareIntColumns time: " << elapsedTime << " milliseconds" << std::endl;
//                       cudaEventDestroy(start);
//                       cudaEventDestroy(stop);
                      
//                   // Copy results back to host
//                   cudaMemcpy(resultVector.data(), d_results, tableSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                  
//                   // Free device memory
//                   cudaFree(d_leftCol);
//                   cudaFree(d_rightCol);
//                   cudaFree(d_results);
//               } else {
//                   // Handle string columns
//                   std::vector<std::string> leftColData(tableSize);
//                   std::vector<std::string> rightColData(tableSize);
                  
//                   // Prepare column data
//                   for (int i = 0; i < tableSize; i++) {
//                       leftColData[i] = leftData[i][leftColIndex];
//                   }
                  
//                   for (int i = 0; i < tableSize; i++) {
//                       rightColData[i] = rightData[i][rightColIndex];
//                   }
                  
//                   // Create array of C-style strings on device
//                   char** h_leftStrings = new char*[tableSize];
//                   char** h_rightStrings = new char*[tableSize];
                  
//                   // Allocate memory for each string on device
//                   for (int i = 0; i < tableSize; i++) {
//                       cudaMalloc(&h_leftStrings[i], leftColData[i].size() + 1);
//                       cudaMemcpy(h_leftStrings[i], leftColData[i].c_str(), 
//                                 leftColData[i].size() + 1, cudaMemcpyHostToDevice);
//                   }
                  
//                   for (int i = 0; i < tableSize; i++) {
//                       cudaMalloc(&h_rightStrings[i], rightColData[i].size() + 1);
//                       cudaMemcpy(h_rightStrings[i], rightColData[i].c_str(), 
//                                 rightColData[i].size() + 1, cudaMemcpyHostToDevice);
//                   }
                  
//                   // Copy arrays of pointers to device
//                   char** d_leftStrings, **d_rightStrings;
//                   uint8_t* d_results;
                  
//                   cudaMalloc(&d_leftStrings, tableSize * sizeof(char*));
//                   cudaMalloc(&d_rightStrings, tableSize * sizeof(char*));
//                   cudaMalloc(&d_results, tableSize * sizeof(uint8_t));
                  
//                   cudaMemcpy(d_leftStrings, h_leftStrings, tableSize * sizeof(char*), cudaMemcpyHostToDevice);
//                   cudaMemcpy(d_rightStrings, h_rightStrings, tableSize * sizeof(char*), cudaMemcpyHostToDevice);
                  
                  
//                   // Launch string comparison kernel
          
//                   cudaEvent_t start, stop;
//                   cudaEventCreate(&start);
//                   cudaEventCreate(&stop);
//                   cudaEventRecord(start);
          
//                   compareStringColumnsOneTable<<<gridDim, blockDim>>>(
//                       d_leftStrings, d_rightStrings, tableSize, tableSize, d_results, opType);
                  
//                   cudaEventRecord(stop);
//                   cudaEventSynchronize(stop);
//                   float elapsedTime;
//                   cudaEventElapsedTime(&elapsedTime, start, stop);
//                   std::cout << "Elapsed time: " << elapsedTime << " milliseconds" << std::endl;
//                   cudaEventDestroy(start);
//                   cudaEventDestroy(stop);
//                   // Copy results back to host
//                   cudaMemcpy(resultVector.data(), d_results, tableSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                  
//                   // Free device memory
//                   for (int i = 0; i < tableSize; i++) {
//                       cudaFree(h_leftStrings[i]);
//                   }
                  
//                   for (int i = 0; i < tableSize; i++) {
//                       cudaFree(h_rightStrings[i]);
//                   }
                  
//                   cudaFree(d_leftStrings);
//                   cudaFree(d_rightStrings);
//                   cudaFree(d_results);
                  
//                   delete[] h_leftStrings;
//                   delete[] h_rightStrings;
//               }
//           }
          
          
          
//               }
//     }
    
//     return resultVector;
// }



// // Main function with both string and integer handling
// std::vector<uint8_t> GPUManager::processComparisonExpr(
//     const Table& leftTable, 
//     const Table& rightTable,
//     const hsql::Expr* expr) 
// {
//     if (!expr || expr->type != hsql::kExprOperator) {
//         throw std::runtime_error("Expected comparison expression");
//     }
    
//     int leftSize = leftTable.getSize();
//     int rightSize = rightTable.getSize();
//     int resultSize = leftSize * rightSize;
    
//     std::vector<uint8_t> resultVector(resultSize, 0);
    
//     // Column-column comparison
//     if (expr->expr->type == hsql::kExprColumnRef && expr->expr2->type == hsql::kExprColumnRef) {
//         const char* leftColName = expr->expr->name;
//         const char* rightColName = expr->expr2->name;
        
//         int leftColIndex = findColumnIndex(leftTable, leftColName, expr->expr->table);
//         int rightColIndex = findColumnIndex(rightTable, rightColName, expr->expr2->table);
        
//         if (leftColIndex == -1 || rightColIndex == -1) {
//             // throw std::runtime_error("Column not found in comparison");
//             return  std::vector<uint8_t> (resultSize, 1);
//         }
        
//         // Determine the operator type
//         int opType;
//         switch (expr->opType) {
//             case hsql::OperatorType::kOpEquals: opType = 0; break;
//             case hsql::OperatorType::kOpNotEquals: opType = 1; break;
//             case hsql::OperatorType::kOpLess: opType = 2; break;
//             case hsql::OperatorType::kOpGreater: opType = 3; break;
//             case hsql::OperatorType::kOpLessEq: opType = 4; break;
//             case hsql::OperatorType::kOpGreaterEq: opType = 5; break;
//             default: throw std::runtime_error("Unsupported operator type");
//         }
        
//         const auto& leftData = leftTable.getData();
//         const auto& rightData = rightTable.getData();
        
//         // Check if we're dealing with integer or string columns
//         bool isIntegerComparison = false;
//         if (!leftData.empty() && !rightData.empty()) {
//             // Sample the first row of each table to determine type
//             isIntegerComparison = isInteger(leftData[0][leftColIndex]) && 
//                                  isInteger(rightData[0][rightColIndex]);
//         }
        
//         // Set up grid and block dimensions for 2D execution
//         dim3 blockDim(16, 16);
//         dim3 gridDim(
//             (leftSize + blockDim.x - 1) / blockDim.x,
//             (rightSize + blockDim.y - 1) / blockDim.y
//         );
        
//         if (isIntegerComparison) {
//             // Handle integer columns
//             std::vector<int> leftColData(leftSize);
//             std::vector<int> rightColData(rightSize);
            
//             // Prepare column data
//             for (int i = 0; i < leftSize; i++) {
//                 leftColData[i] = std::stoi(leftData[i][leftColIndex]);
//             }
            
//             for (int i = 0; i < rightSize; i++) {
//                 rightColData[i] = std::stoi(rightData[i][rightColIndex]);
//             }
            
//             // Allocate device memory
//             int *d_leftCol, *d_rightCol;
//             uint8_t *d_results;
            
//             cudaMalloc(&d_leftCol, leftSize * sizeof(int));
//             cudaMalloc(&d_rightCol, rightSize * sizeof(int));
//             cudaMalloc(&d_results, resultSize * sizeof(uint8_t));
            
//             // Copy data to device
//             cudaMemcpy(d_leftCol, leftColData.data(), leftSize * sizeof(int), cudaMemcpyHostToDevice);
//             cudaMemcpy(d_rightCol, rightColData.data(), rightSize * sizeof(int), cudaMemcpyHostToDevice);
            
//             cudaEvent_t start, stop;
//             cudaEventCreate(&start);
//             cudaEventCreate(&stop);
//             cudaEventRecord(start);
//             // Launch integer comparison kernel
//             // compareIntColumns<<<gridDim, blockDim>>>(
//             //     d_leftCol, d_rightCol, leftSize, rightSize, d_results, opType);
            
//                 cudaEventRecord(stop);
//                 cudaEventSynchronize(stop);
//                 float elapsedTime;
//                 cudaEventElapsedTime(&elapsedTime, start, stop);
//                 std::cout << "Elapsed compareIntColumns time: " << elapsedTime << " milliseconds" << std::endl;
//                 cudaEventDestroy(start);
//                 cudaEventDestroy(stop);
                
//             // Copy results back to host
//             cudaMemcpy(resultVector.data(), d_results, resultSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            
//             // Free device memory
//             cudaFree(d_leftCol);
//             cudaFree(d_rightCol);
//             cudaFree(d_results);
//         } else {
//             // Handle string columns
//             std::vector<std::string> leftColData(leftSize);
//             std::vector<std::string> rightColData(rightSize);
            
//             // Prepare column data
//             for (int i = 0; i < leftSize; i++) {
//                 leftColData[i] = leftData[i][leftColIndex];
//             }
            
//             for (int i = 0; i < rightSize; i++) {
//                 rightColData[i] = rightData[i][rightColIndex];
//             }
            
//             // Create array of C-style strings on device
//             char** h_leftStrings = new char*[leftSize];
//             char** h_rightStrings = new char*[rightSize];
            
//             // Allocate memory for each string on device
//             for (int i = 0; i < leftSize; i++) {
//                 cudaMalloc(&h_leftStrings[i], leftColData[i].size() + 1);
//                 cudaMemcpy(h_leftStrings[i], leftColData[i].c_str(), 
//                           leftColData[i].size() + 1, cudaMemcpyHostToDevice);
//             }
            
//             for (int i = 0; i < rightSize; i++) {
//                 cudaMalloc(&h_rightStrings[i], rightColData[i].size() + 1);
//                 cudaMemcpy(h_rightStrings[i], rightColData[i].c_str(), 
//                           rightColData[i].size() + 1, cudaMemcpyHostToDevice);
//             }
            
//             // Copy arrays of pointers to device
//             char** d_leftStrings, **d_rightStrings;
//             uint8_t* d_results;
            
//             cudaMalloc(&d_leftStrings, leftSize * sizeof(char*));
//             cudaMalloc(&d_rightStrings, rightSize * sizeof(char*));
//             cudaMalloc(&d_results, resultSize * sizeof(uint8_t));
            
//             cudaMemcpy(d_leftStrings, h_leftStrings, leftSize * sizeof(char*), cudaMemcpyHostToDevice);
//             cudaMemcpy(d_rightStrings, h_rightStrings, rightSize * sizeof(char*), cudaMemcpyHostToDevice);
            
            
//             // Launch string comparison kernel

//             cudaEvent_t start, stop;
//             cudaEventCreate(&start);
//             cudaEventCreate(&stop);
//             cudaEventRecord(start);

//             compareStringColumns<<<gridDim, blockDim>>>(
//                 d_leftStrings, d_rightStrings, leftSize, rightSize, d_results, opType);
            
//             cudaEventRecord(stop);
//             cudaEventSynchronize(stop);
//             float elapsedTime;
//             cudaEventElapsedTime(&elapsedTime, start, stop);
//             std::cout << "Elapsed time: " << elapsedTime << " milliseconds" << std::endl;
//             cudaEventDestroy(start);
//             cudaEventDestroy(stop);
//             // Copy results back to host
//             cudaMemcpy(resultVector.data(), d_results, resultSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            
//             // Free device memory
//             for (int i = 0; i < leftSize; i++) {
//                 cudaFree(h_leftStrings[i]);
//             }
            
//             for (int i = 0; i < rightSize; i++) {
//                 cudaFree(h_rightStrings[i]);
//             }
            
//             cudaFree(d_leftStrings);
//             cudaFree(d_rightStrings);
//             cudaFree(d_results);
            
//             delete[] h_leftStrings;
//             delete[] h_rightStrings;
//         }
//     }
    
//     return resultVector;
// }

// std::vector<uint8_t> GPUManager::processBinaryExpr(
//     const Table& leftTable, 
//     const Table& rightTable,
//     const hsql::Expr* expr) 
// {
//     // This is a simplified implementation
//     if (expr->type == hsql::kExprOperator) {
//         if (expr->opType == hsql::OperatorType::kOpAnd || expr->opType == hsql::OperatorType::kOpOr) {
//               // Process binary kOpAnd/kOpOr operations
//               auto leftResults = processBinaryExpr(leftTable, rightTable, expr->expr);
//               auto rightResults = processBinaryExpr(leftTable, rightTable, expr->expr2);
              
//               int resultSize = leftTable.getSize() * rightTable.getSize();
              
//               // Create device vectors
//               uint8_t *d_leftResults, *d_rightResults, *d_output;
//               cudaMalloc(&d_leftResults, resultSize * sizeof(uint8_t));
//               cudaMalloc(&d_rightResults, resultSize * sizeof(uint8_t));
//               cudaMalloc(&d_output, resultSize * sizeof(uint8_t));
              
//               // Copy data to device
//               cudaMemcpy(d_leftResults, leftResults.data(), resultSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
//               cudaMemcpy(d_rightResults, rightResults.data(), resultSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
              
//               // Set up kernel execution parameters
//               int blockSize = 256;
//               int numBlocks = (resultSize + blockSize - 1) / blockSize;
              
//               // Execute kernel
//               uint8_t isAnd = expr->opType == hsql::OperatorType::kOpAnd ? 1 : 0;
//               combineResults<<<numBlocks, blockSize>>>(d_leftResults, d_rightResults, d_output, resultSize, isAnd);
              
//               // Copy results back to host
//               std::vector<uint8_t> resultVector(resultSize);
//               cudaMemcpy(resultVector.data(), d_output, resultSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
              
//               // Free device memory
//               cudaFree(d_leftResults);
//               cudaFree(d_rightResults);
//               cudaFree(d_output);
              
//               return resultVector;
//         } else {
//             return processComparisonExpr(leftTable, rightTable, expr);
//         }
//     }
    
//     // Default case - no conditions
//     int resultSize = leftTable.getSize() * rightTable.getSize();
//     return std::vector<uint8_t>(resultSize, 1);  // 1 means true
// }



///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////


