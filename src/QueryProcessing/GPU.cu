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
const int GPUManager::BATCH_SIZE = 5000;
#define BLOCK_SIZE_KOGGE_STONE 1024
int64_t *d_results_join;

// =========== KOGGE-STONE ALGORITHM (WORK INEFFICIENT) ===========
// __global__ void koggeStone(int64_t *d_data, int64_t *d_output, int64_t *partialSums, int n)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx >= n)
//         return;

//     // using double buffering
//     __shared__ int64_t buffer1_s[BLOCK_SIZE_KOGGE_STONE];
//     __shared__ int64_t buffer2_s[BLOCK_SIZE_KOGGE_STONE];
//     int64_t *inBuffer_s = buffer1_s;
//     int64_t *outBuffer_s = buffer2_s;

//     // Load data into shared memory
//     inBuffer_s[threadIdx.x] = d_data[idx];
//     __syncthreads();

//     // do prefix sum on the block
//     for (int stride = 1; stride <= BLOCK_SIZE_KOGGE_STONE / 2; stride *= 2)
//     {
//         if (threadIdx.x >= stride)
//             outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
//         else
//             outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
//         __syncthreads();
//         int64_t *tempbuffer = inBuffer_s;
//         inBuffer_s = outBuffer_s;
//         outBuffer_s = tempbuffer;
//     }

//     // save to partial sums
//     if (threadIdx.x == blockDim.x - 1)
//     {
//         partialSums[blockIdx.x] = inBuffer_s[threadIdx.x];
//     }

//     // save to output
//     d_output[idx] = inBuffer_s[threadIdx.x];
// }

// __global__ void handlePartialSumKogge(int64_t *output, int64_t *partialSums, int n)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= n)
//         return;

//     if (blockIdx.x > 0)
//     {
//         // add patial sum to the block
//         output[idx] += partialSums[blockIdx.x - 1];
//     }
// }

// void koggeStoneCPU(int64_t *d_data, int64_t *d_result, int n)
// {
//     int64_t *d_partialSums;
//     int numBlocks = (n + BLOCK_SIZE_KOGGE_STONE - 1) / BLOCK_SIZE_KOGGE_STONE;
//     cudaMalloc(&d_partialSums, numBlocks * sizeof(int64_t));

//     koggeStone<<<numBlocks, BLOCK_SIZE_KOGGE_STONE>>>(d_data, d_result, d_partialSums, n);
//     cudaDeviceSynchronize();

//     if (numBlocks > 1)
//     {
//         koggeStoneCPU(d_partialSums, d_partialSums, numBlocks);
//         handlePartialSumKogge<<<numBlocks, BLOCK_SIZE_KOGGE_STONE>>>(d_result, d_partialSums, n);
//         cudaDeviceSynchronize();
//     }
//     cudaFree(d_partialSums);
// }


__global__ void efficient_prefix_sum(int64_t* input, int64_t* output, int n, int64_t* aux) {
    extern __shared__ int64_t temp[];

    int64_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int64_t t = threadIdx.x;

    temp[t]              = 0;
    temp[t + blockDim.x] = 0;

    if (idx < n)              temp[t]              = input[idx];
    if (idx + blockDim.x < n) temp[t + blockDim.x] = input[idx + blockDim.x];

    int64_t factor = 1;
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            const int64_t ai = factor * ( 2 * t + 1 ) - 1;
            const int64_t bi = factor * ( 2 * t + 2 ) - 1;
            temp[bi] += temp[ai];
        }
        factor <<= 1;
    }

    __syncthreads();

    if (t == 0) {
        temp[blockDim.x * 2 - 1] = 0;
    }

    factor = 1;
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        __syncthreads();

        if (t < factor) {
            const int64_t ai = stride * ( 2 * t + 1 ) - 1;
            const int64_t bi = stride * ( 2 * t + 2 ) - 1;
            const int64_t val = temp[ai];

            temp[ai]  = temp[bi];
            temp[bi] += val;
        }

        factor <<= 1;
    }

    __syncthreads();

    if (t == 0 && aux != nullptr) aux[blockIdx.x] = temp[blockDim.x * 2 - 1] + input[blockIdx.x * blockDim.x * 2 + blockDim.x * 2 - 1];

    __syncthreads();

    if (idx < n)              output[idx]              = temp[t] + input[idx];
    if (idx + blockDim.x < n) output[idx + blockDim.x] = temp[t + blockDim.x] + input[idx + blockDim.x];
}

__global__ void efficient_prefix_sum(char* input, int64_t* output, int n, int64_t* aux) {
    extern __shared__ int64_t temp[];

    int64_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int64_t t = threadIdx.x;

    temp[t]              = 0;
    temp[t + blockDim.x] = 0;

    if (idx < n)              temp[t]              = input[idx];
    if (idx + blockDim.x < n) temp[t + blockDim.x] = input[idx + blockDim.x];

    int64_t factor = 1;
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            const int64_t ai = factor * ( 2 * t + 1 ) - 1;
            const int64_t bi = factor * ( 2 * t + 2 ) - 1;
            temp[bi] += temp[ai];
        }
        factor <<= 1;
    }

    __syncthreads();

    if (t == 0) {
        temp[blockDim.x * 2 - 1] = 0;
    }

    factor = 1;
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        __syncthreads();

        if (t < factor) {
            const int64_t ai = stride * ( 2 * t + 1 ) - 1;
            const int64_t bi = stride * ( 2 * t + 2 ) - 1;
            const int64_t val = temp[ai];

            temp[ai]  = temp[bi];
            temp[bi] += val;
        }

        factor <<= 1;
    }

    __syncthreads();

    if (t == 0 && aux != nullptr) aux[blockIdx.x] = temp[blockDim.x * 2 - 1] + input[blockIdx.x * blockDim.x * 2 + blockDim.x * 2 - 1];

    __syncthreads();

    if (idx < n)              output[idx]              = temp[t] + input[idx];
    if (idx + blockDim.x < n) output[idx + blockDim.x] = temp[t + blockDim.x] + input[idx + blockDim.x];
}

__global__ void add_aux(int64_t* input, int n, const int64_t* aux) {
    int64_t idx = (blockIdx.x + 1) * blockDim.x * 2 + threadIdx.x;

    if   (idx >= n) return;
    input[idx]              = aux[blockIdx.x] + input[idx];

    if   (idx + blockDim.x >= n) return;
    input[idx + blockDim.x] = aux[blockIdx.x] + input[idx + blockDim.x];
}

static void run_scan(int64_t* input, int64_t* output, int64_t n) {
    if (n == 0) return;

    int64_t blocks = (n + 256 * 2 - 1) / (256 * 2);

    int64_t *d_block_sums;
    cudaMalloc(&d_block_sums, blocks * sizeof(int64_t));

    efficient_prefix_sum<<<blocks, 256, (256 * 2 + 1) * sizeof(int64_t)>>>(input, output, n, d_block_sums);
    // CUDA_CHECK_LAST_ERROR("efficient_prefix_sum");

    if (blocks > 256) {
        int64_t* r_v;
        cudaMalloc(&r_v, blocks * sizeof(int64_t));

        run_scan(d_block_sums, r_v, blocks);
        // CUDA_CHECK_LAST_ERROR("run_scan::recursive");
        cudaFree(d_block_sums);
        d_block_sums = r_v;
    } else {
        efficient_prefix_sum<<<1, 256, (256 * 2 + 1) * sizeof(int64_t)>>>(d_block_sums, d_block_sums, blocks, nullptr);
        // CUDA_CHECK_LAST_ERROR("efficient_prefix_sum");
    }

    if (blocks > 1) {
        // CUDA_CHECK_LAST_ERROR("add_aux before");
        add_aux<<<blocks - 1, 256>>>(output, n, d_block_sums);
        // CUDA_CHECK_LAST_ERROR("add_aux");
    }

    cudaFree(d_block_sums);
}

static void run_scan(char* input, int64_t* output, int64_t n) {
    if (n == 0) return;

    int64_t blocks = (n + 256 * 2 - 1) / (256 * 2);

    int64_t* d_block_sums;
    cudaMalloc(&d_block_sums, blocks * sizeof(int64_t));


    efficient_prefix_sum<<<blocks, 256, (256 * 2) * sizeof(int64_t)>>>(input, output, n, d_block_sums);
    // CUDA_CHECK_LAST_ERROR("efficient_prefix_sum");

    if (blocks > 256) {
        int64_t* r_v;

        cudaMalloc(&r_v, blocks * sizeof(int64_t));

        run_scan(d_block_sums, r_v, blocks);
        // CUDA_CHECK_LAST_ERROR("run_scan::recursive");
        cudaFree(d_block_sums);
        d_block_sums = r_v;
    } else {
        efficient_prefix_sum<<<1, 256, (256 * 2) * sizeof(int64_t)>>>(d_block_sums, d_block_sums, blocks, nullptr);
        // CUDA_CHECK_LAST_ERROR("efficient_prefix_sum");
    }

    if (blocks > 1) {

        add_aux<<<blocks - 1, 256>>>(output, n, d_block_sums);
        // CUDA_CHECK_LAST_ERROR("add_aux");
    }

    cudaFree(d_block_sums);
}

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
    int64_t* __restrict__ results,
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
            int64_t match;
            
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
    const int64_t* results1, 
    const int64_t* results2,
    int64_t* output, 
    int size, 
    int64_t isAnd) 
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
    for (int64_t col = 0; col < headers.size(); col++) {
        columnData[headers[col]] = std::vector<unionV>(resultData.size());
    }
    
    // Populate column data
    for (int64_t row = 0; row < resultData.size(); row++) {
        for (int64_t col = 0; col < headers.size(); col++) {
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
std::vector<int64_t> GPUManager::processBatch(
    const std::vector<std::shared_ptr<Table>>& tables,
    const std::vector<std::vector<int>>& batchIndices,
    const hsql::Expr* conditions) {
    
    // Calculate batch size
    int batchSize = 1;
    for (const auto& indices : batchIndices) {
        batchSize *= indices.size();
    }
    
    // Initialize all results to 1 (true)
    std::vector<int64_t> results(batchSize, 1);
    
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
            int64_t *d_leftResults, *d_rightResults, *d_output;
            cudaMalloc(&d_leftResults, batchSize * sizeof(int64_t));
            cudaMalloc(&d_rightResults, batchSize * sizeof(int64_t));
            cudaMalloc(&d_output, batchSize * sizeof(int64_t));
            
            cudaMemcpy(d_leftResults, leftResults.data(), batchSize * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rightResults, rightResults.data(), batchSize * sizeof(int64_t), cudaMemcpyHostToDevice);
            
            int blockSize = 256;
            int numBlocks = (batchSize + blockSize - 1) / blockSize;
            
            int64_t isAnd = conditions->opType == hsql::OperatorType::kOpAnd ? 1 : 0;
            combineResults<<<numBlocks, blockSize>>>(d_leftResults, d_rightResults, d_output, batchSize, isAnd);
            
            cudaMemcpy(results.data(), d_output, batchSize * sizeof(int64_t), cudaMemcpyDeviceToHost);
            
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
std::vector<int64_t> GPUManager::evaluateConditionOnBatch(
    const std::vector<std::shared_ptr<Table>>& tables,
    const std::vector<std::vector<int>>& batchIndices,
    const hsql::Expr* condition) {
    
    // Calculate total batch size (cartesian product of all tables)
    int batchSize = 1;
    for (const auto& indices : batchIndices) {
        batchSize *= indices.size();
    }
    
    // Default to all true
    std::vector<int64_t> results(batchSize, 1);
    
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
        int64_t *d_results;
        int* d_tableSizes;
        
        cudaMalloc(&d_leftCol, leftBatchSize * sizeof(int64_t));
        cudaMalloc(&d_rightCol, rightBatchSize * sizeof(int64_t));
        cudaMalloc(&d_results, batchSize * sizeof(int64_t));
        
        // Initialize results to 1 (true)
        cudaMemsetAsync(d_results, 1, batchSize * sizeof(int64_t), stream);
        
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
        cudaMemcpyAsync(results.data(), d_results, batchSize * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        
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


//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

// New kernel for direct two-table join evaluation

__global__ void compareIntColumns(
    const int64_t*  leftColumn,
    const int64_t*  rightColumn,
    int leftSize, 
    int rightSize,
    int64_t* results, 
    int opType,
    const int* tableSizes,  // Array containing sizes of all tables
    int numTables,          // Total number of tables
    int leftTableIdx,       // Index of left table in the tables array
    int rightTableIdx)      // Index of right table in the tables array
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < leftSize && j < rightSize) {
        int64_t match = 0;
        
        // Compare the values based on the operation type
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
        
        int flatIndex = i * tableSizes[1] + j;
        results[flatIndex] = match;
    }
}


__global__ void evaluateTwoTableJoin(
    const int64_t* __restrict__ leftColumn,
    const int64_t* __restrict__ rightColumn,
    int leftTableSize,
    int rightTableSize,
    int opType,
    int64_t* __restrict__ results)
{
    // Each thread processes multiple combinations for better efficiency
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int totalCombinations = leftTableSize * rightTableSize;
    
    // Process combinations in stride
    for (int i = idx; i < totalCombinations; i += stride) {
        // Calculate left and right indices from the flattened index
        int leftIdx = i / rightTableSize;
        int rightIdx = i % rightTableSize;
        
        // Get values to compare
        int64_t leftValue = leftColumn[leftIdx];
        int64_t rightValue = rightColumn[rightIdx];
        
        // Evaluate the comparison (branch-free implementation for less divergence)
        int64_t match;
        
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
        results[i] = match;
    }
}


// Apply block sums to get the final prefix sum
__global__ void addBlockSums(int* output, int* blockSums, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && blockIdx.x > 0) {
        output[i] += blockSums[blockIdx.x - 1];
    }
}

std::vector<int64_t> iterator(int64_t* cpu_out, int size ) {
    std::vector<int64_t> result;

    int64_t prev = 0;

    for (int64_t i = 0;i < size;i++) {
        auto val = cpu_out[i];
        if (val == prev) {
            auto low = i;
            auto high = size;
            while (low < high) {
                auto mid = low + (high - low) / 2;
                if (cpu_out[mid] == val) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }

            i = low - 1;
        } else {
            result.push_back(i);
            prev = val;
        }
    }

    return result;
}

// Function to perform binary search on prefix sum array to find match positions
__global__ void binarySearchMatches(
    const int64_t*  prefixSum,
    int*  leftIndices,
    int*  rightIndices,
    int totalSize,
    int rightTableSize,
    int matchCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < matchCount) {
        // Binary search to find the position where prefixSum[pos] >= idx+1
        // This gives us the position in the original results array
        int low = 0;
        int high = totalSize - 1;
        int pos = -1;
        
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (prefixSum[mid] >= idx + 1) {
                pos = mid;
                high = mid - 1; // Continue searching left for leftmost occurrence
            } else {
                low = mid + 1;
            }
        }
        
        if (pos != -1) {
            // Calculate original row indices
            leftIndices[idx] = pos / rightTableSize;
            rightIndices[idx] = pos % rightTableSize;
        }
    }
}


// Method to evaluate join condition directly between two tables
std::vector<int64_t> GPUManager::evaluateTwoTableJoinCondition(
    const std::shared_ptr<Table>& leftTable,
    const std::shared_ptr<Table>& rightTable,
    hsql::Expr* condition)
{
    int leftSize = leftTable->getSize();
    int rightSize = rightTable->getSize();
    std::vector<int> tableSizes;
    tableSizes.push_back(leftSize);
    tableSizes.push_back(rightSize);
    int totalCombinations = leftSize * rightSize;
    
    // Default to all true
    std::vector<int64_t> results(totalCombinations, 1);
    
    // Handle AND/OR conditions
    if (condition->type == hsql::kExprOperator && 
        (condition->opType == hsql::OperatorType::kOpAnd || 
         condition->opType == hsql::OperatorType::kOpOr)) {
        
        // Process left and right conditions separately
        auto leftResults = evaluateTwoTableJoinCondition(leftTable, rightTable, condition->expr);
        auto rightResults = evaluateTwoTableJoinCondition(leftTable, rightTable, condition->expr2);

        // Combine results on GPU
        int64_t *d_leftResults, *d_rightResults;
        cudaMalloc(&d_leftResults, totalCombinations * sizeof(int64_t));
        cudaMalloc(&d_rightResults, totalCombinations * sizeof(int64_t));
        
        cudaMemcpy(d_leftResults, leftResults.data(), totalCombinations * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rightResults, rightResults.data(), totalCombinations * sizeof(int64_t), cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int numBlocks = (totalCombinations + blockSize - 1) / blockSize;
        
        int64_t isAnd = condition->opType == hsql::OperatorType::kOpAnd ? 1 : 0;
        combineResults<<<numBlocks, blockSize>>>(d_leftResults, d_rightResults, d_results_join, totalCombinations, isAnd);
        
        cudaMemcpy(results.data(), d_results_join, totalCombinations * sizeof(int64_t), cudaMemcpyDeviceToHost);
    

        cudaFree(d_leftResults);
        cudaFree(d_rightResults);
        
        return results;
    }
    
    // Handle comparison operator between columns
    hsql::printExpression(condition, 5);
    if (condition->type == hsql::kExprOperator && 
        condition->expr->type == hsql::kExprColumnRef && 
        condition->expr2->type == hsql::kExprColumnRef) {
        
        // Extract column names
        std::string leftColName = condition->expr->name;
        std::string rightColName = condition->expr2->name;
        
        // Check if we need to prepend table names/aliases
        std::string leftTableName = condition->expr->table ? condition->expr->table : "";
        std::string rightTableName = condition->expr2->table ? condition->expr2->table : "";
        
        // Find column indices
        int leftColIdx = findColumnIndex(*leftTable, leftColName.c_str(), leftTableName.c_str());
        int rightColIdx = findColumnIndex(*rightTable, rightColName.c_str(), rightTableName.c_str());
        
        // If columns not found or aren't in the expected tables, try swapping
        if (leftColIdx == -1 || rightColIdx == -1) {
            leftColIdx = findColumnIndex(*leftTable, rightColName.c_str(), rightTableName.c_str());
            rightColIdx = findColumnIndex(*rightTable, leftColName.c_str(), leftTableName.c_str());
            
            // If still not found, return all true
            if (leftColIdx == -1 || rightColIdx == -1) {
                return results;
            }
            
            // Swap operator direction if columns were swapped
            switch (condition->opType) {
                case hsql::OperatorType::kOpLess: 
                    condition->opType = hsql::OperatorType::kOpGreater; 
                    break;
                case hsql::OperatorType::kOpGreater: 
                    condition->opType = hsql::OperatorType::kOpLess; 
                    break;
                case hsql::OperatorType::kOpLessEq: 
                    condition->opType = hsql::OperatorType::kOpGreaterEq; 
                    break;
                case hsql::OperatorType::kOpGreaterEq: 
                    condition->opType = hsql::OperatorType::kOpLessEq; 
                    break;
                default: 
                    break; // No change for equals/not equals
            }
        }
        
        // Get actual column names from the tables
        leftColName = leftTable->getHeaders()[leftColIdx];
        rightColName = rightTable->getHeaders()[rightColIdx];
        
        // Verify column types are integers (current implementation only handles integers)
        if (leftTable->getColumnType(leftColName) != ColumnType::INTEGER || 
            rightTable->getColumnType(rightColName) != ColumnType::INTEGER) {
            // Only implementing integer comparisons in this version
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
        
      // Get integer column data directly (column-major format)
        const auto& leftColumnData = leftTable->getData().at(leftColName);
        const auto& rightColumnData = rightTable->getData().at(rightColName);


        // Allocate GPU memory
        int64_t *d_leftColumn, *d_rightColumn;
        
        int *d_tableSizes;  // Add this line

        cudaMalloc(&d_leftColumn, leftSize * sizeof(int64_t));
        cudaMalloc(&d_rightColumn, rightSize * sizeof(int64_t));
        cudaMalloc(&d_tableSizes, tableSizes.size() * sizeof(int));  // Add this line

        // Copy data to GPU
        cudaMemcpy(d_leftColumn, leftColumnData.data(), leftSize * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rightColumn, rightColumnData.data(), rightSize * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tableSizes, tableSizes.data(), tableSizes.size() * sizeof(int), cudaMemcpyHostToDevice);  // Add this line

        // Calculate grid and block dimensions
        int blockSize = 256;
        int numBlocks = std::min(65535, (totalCombinations + blockSize - 1) / blockSize);

        // Launch kernel with 2D grid/block configuration
        dim3 blockDim(16, 16);
        dim3 gridDim(
            (leftSize + blockDim.x - 1) / blockDim.x,
            (rightSize + blockDim.y - 1) / blockDim.y
        );

        compareIntColumns<<<gridDim, blockDim>>>(
            d_leftColumn, d_rightColumn, 
            leftSize, rightSize, 
            d_results_join, opType, d_tableSizes, 2, 0, 1);  // Use d_tableSizes instead of tableSizes.data()
        
        // Copy results back to host
        // cudaMemcpy(results.data(), d_results, totalCombinations * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(results.data(), d_results_join, results.size() * sizeof(int64_t), cudaMemcpyDeviceToHost);  // Add this line
    
        // Free GPU memory
        cudaFree(d_leftColumn);
        cudaFree(d_rightColumn);
        // cudaFree(d_results);
        cudaFree(d_tableSizes);  // Add this line
    }
    
    return results;
}





// Modified two-table join method that uses binary search on prefix sum
std::shared_ptr<Table> GPUManager::executeTwoTableJoinWithBinarySearch(
    const std::shared_ptr<Table>& leftTable,
    const std::shared_ptr<Table>& rightTable,
    hsql::Expr* joinCondition)
{
    // if (!hasGPU_) {
    //     throw std::runtime_error("GPU operations not available");
    // }
    
    // Calculate total combinations (cross-product size)
    int leftSize = leftTable->getSize();
    int rightSize = rightTable->getSize();


    int totalCombinations = leftSize * rightSize;
    
    // Default to all matches if no condition
    std::vector<int64_t> results(totalCombinations, 1);
    cudaMalloc(&d_results_join, totalCombinations * sizeof(int64_t));

    // Process join condition if one exists
    if (joinCondition) {
        results = evaluateTwoTableJoinCondition(leftTable, rightTable, joinCondition);
    }

    // Device memory pointers
    // int64_t* d_results;
    int64_t* d_prefixSum;
    int* d_blockSums;
    
    // Allocate device memory
    // cudaMalloc(&d_results, totalCombinations * sizeof(int64_t));
    cudaMalloc(&d_prefixSum, totalCombinations * sizeof(int64_t));
    
    // Copy results to device
    // cudaMemcpy(d_results, results.data(), totalCombinations * sizeof(int64_t), cudaMemcpyHostToDevice);
    
    // Calculate grid and block dimensions
    int blockSize = 512; // Power of 2 for efficient scan
    int numBlocks = (totalCombinations + 2 * blockSize - 1) / (2 * blockSize); // Each thread processes 2 elements
    
    
    int64_t *h_result = (int64_t *)malloc(totalCombinations * sizeof(int64_t));

    int64_t *d_result;
    cudaMalloc(&d_result, totalCombinations * sizeof(int64_t));


    // koggeStoneCPU(d_results_join, d_result, totalCombinations);
    run_scan(d_results_join, d_result, totalCombinations);

    cudaMemcpy(h_result, d_result, totalCombinations * sizeof(int64_t), cudaMemcpyDeviceToHost);


    std::vector<int64_t> match_indecies = iterator(h_result, totalCombinations);
    // for (int i=0;i<1000;i++){
    //     std::cout << h_result[i] << '\n';
    // }


    
    // Get total match count (last element of prefix sum + last result)
    // int lastPrefixSum = 0;
    // int64_t lastResult = 0;
    // cudaMemcpy(&lastPrefixSum, &d_result[totalCombinations-1], sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&lastResult, &d_results_join[totalCombinations-1], sizeof(int64_t), cudaMemcpyDeviceToHost);
    // int matchCount = lastPrefixSum + lastResult;
    
    // // Allocate memory for match indices
    // int* d_leftIndices;
    // int* d_rightIndices;
    // cudaMalloc(&d_leftIndices, matchCount * sizeof(int));
    // cudaMalloc(&d_rightIndices, matchCount * sizeof(int));
    
    // // Find match positions using binary search on prefix sum
    // int bsBlockSize = 256;
    // int bsNumBlocks = (matchCount + bsBlockSize - 1) / bsBlockSize;
    
    // binarySearchMatches<<<bsNumBlocks, bsBlockSize>>>(
    //     d_result, d_leftIndices, d_rightIndices, 
    //     totalCombinations, rightSize, matchCount);
    
    // // Copy indices back to host
    // std::vector<int> leftIndices(matchCount);
    // std::vector<int> rightIndices(matchCount);
    // cudaMemcpy(leftIndices.data(), d_leftIndices, matchCount * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(rightIndices.data(), d_rightIndices, matchCount * sizeof(int), cudaMemcpyDeviceToHost);
    
    // // Free GPU memory
    // // cudaFree(d_results);
    // cudaFree(d_prefixSum);
    // cudaFree(d_blockSums);
    // cudaFree(d_leftIndices);
    // cudaFree(d_rightIndices);
    
    // Prepare and populate the result table (same as before)
    // Combine headers from both tables
    std::vector<std::string> headers;
    std::unordered_map<std::string, ColumnType> columnTypes;
    std::unordered_map<std::string, std::vector<unionV>> columnData;
    
    // Add left table headers
    const auto& leftHeaders = leftTable->getHeaders();
    const auto& leftTypes = leftTable->getColumnTypes();
    const std::string& leftAlias = leftTable->getAlias();
    
    for (const auto& header : leftHeaders) {
        std::string qualifiedName = leftAlias.empty() ? header : leftAlias + "." + header;
        headers.push_back(qualifiedName);
        columnTypes[qualifiedName] = leftTypes.at(header);
    }
    
    // Add right table headers
    const auto& rightHeaders = rightTable->getHeaders();
    const auto& rightTypes = rightTable->getColumnTypes();
    const std::string& rightAlias = rightTable->getAlias();
    
    for (const auto& header : rightHeaders) {
        std::string qualifiedName = rightAlias.empty() ? header : rightAlias + "." + header;
        headers.push_back(qualifiedName);
        columnTypes[qualifiedName] = rightTypes.at(header);
    }
    
    // Initialize column data structure with correct size
    for (const auto& header : headers) {
        columnData[header] = std::vector<unionV>(match_indecies.size());
    }
    
    // Populate result table using the matching indices
    for (int resultIdx = 0; resultIdx < match_indecies.size(); resultIdx++) {
        int leftIdx = match_indecies[resultIdx] / rightSize;
        int rightIdx = match_indecies[resultIdx] % rightSize;
        
        // Add data from left table
        auto leftRow = leftTable->getRow(leftIdx);
        auto rightRow = rightTable->getRow(rightIdx);
        
        // Add data from left table
        for (int64_t colIdx = 0; colIdx < leftHeaders.size(); ++colIdx) {
            const auto& header = leftHeaders[colIdx];
            std::string qualifiedName = leftAlias.empty() ? header : leftAlias + "." + header;
            columnData[qualifiedName][resultIdx] = leftRow[colIdx];
        }
        
        // Add data from right table
        for (int64_t colIdx = 0; colIdx < rightHeaders.size(); ++colIdx) {
            const auto& header = rightHeaders[colIdx];
            std::string qualifiedName = rightAlias.empty() ? header : rightAlias + "." + header;
            columnData[qualifiedName][resultIdx] = rightRow[colIdx];
        }
    }
    cudaFree(d_results_join);
    // Create and return the joined table
    return std::make_shared<Table>("joined_result", headers, columnData, columnTypes);
}










//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////


// // Updated kernel to handle full batch size
// __global__ void evaluateComparisonBatch(
//     const int* leftColumn,
//     const int* rightColumn,
//     const int* tableSizes,
//     int numTables,
//     int leftTableIdx,
//     int rightTableIdx,
//     int opType,
//     int64_t* results,
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
//         int64_t match = 0;
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
//     int64_t* results, 
//     int opType) 
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
    
//     if (i < leftSize && j < rightSize) {
//         int64_t match = 0;
        
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
//    int64_t* results, 
//    int opType) 
// {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
   
//    if (i < leftSize) {
//        int64_t match = 0;
       
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
//     int64_t* results, 
//     int opType,
//     const int* tableSizes,  // Array containing sizes of all tables
//     int numTables,          // Total number of tables
//     int leftTableIdx,       // Index of left table in the tables array
//     int rightTableIdx)      // Index of right table in the tables array
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
    
//     if (i < leftSize && j < rightSize) {
//         int64_t match = 0;
        
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
//    int64_t* results, 
//    int opType) 
// {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
   
//    if (i < leftSize) {
//        int64_t match = 0;
       
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
//     int64_t* results, 
//     int opType) 
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (i < size) {
//         int64_t match = 0;
        
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
//     int64_t* results,
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
        
//         int64_t match = 0;
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




// std::vector<int64_t> GPUManager::gpuJoinTables(
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
    
//     std::vector<int64_t> resultVector(resultSize, 0);
    
//     // Process each condition and combine results
//     if (conditions->type == hsql::kExprOperator) {
//         if (conditions->opType == hsql::OperatorType::kOpAnd || conditions->opType == hsql::OperatorType::kOpOr) {
//             // Process binary kOpAnd/kOpOr operations
//             auto leftResults = processBinaryExpr(leftTable, rightTable, conditions->expr);
//             auto rightResults = processBinaryExpr(leftTable, rightTable, conditions->expr2);
            
//             // Create device vectors
//             int64_t *d_leftResults, *d_rightResults, *d_output;
//             cudaMalloc(&d_leftResults, resultSize * sizeof(int64_t));
//             cudaMalloc(&d_rightResults, resultSize * sizeof(int64_t));
//             cudaMalloc(&d_output, resultSize * sizeof(int64_t));
            
//             // Copy data to device
//             cudaMemcpy(d_leftResults, leftResults.data(), resultSize * sizeof(int64_t), cudaMemcpyHostToDevice);
//             cudaMemcpy(d_rightResults, rightResults.data(), resultSize * sizeof(int64_t), cudaMemcpyHostToDevice);
            
//             // Set up kernel execution parameters
//             int blockSize = 256;
//             int numBlocks = (resultSize + blockSize - 1) / blockSize;
            
//             // Execute kernel
//             int64_t isAnd = conditions->opType == hsql::OperatorType::kOpAnd ? 1 : 0;
//             combineResults<<<numBlocks, blockSize>>>(d_leftResults, d_rightResults, d_output, resultSize, isAnd);
            
//             // Copy results back to host
//             cudaMemcpy(resultVector.data(), d_output, resultSize * sizeof(int64_t), cudaMemcpyDeviceToHost);
            
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
//                                              const std::vector<int64_t>& mask) 
// {
//     return std::make_shared<Table>(
//         table.getName() + "_filtered",
//         table.getHeaders(),
//         mergeFilterResults(table, mask)
//     );
// }

// std::vector<std::vector<std::string>> GPUManager::mergeFilterResults(
//     const Table& table,
//     const std::vector<int64_t>& mask) const 
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
//     const std::vector<int64_t>& mask) const {
    
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

// std::vector<int64_t> GPUManager::gpuFilterTable(
//     const Table& table,
//     const hsql::Expr* conditions) 
// {
//     if (!hasGPU_) {
//         throw std::runtime_error("GPU operations not available");
//     }

    
//     int tableSize = table.getSize();
//     std::vector<int64_t> resultVector(tableSize, 0);
    
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
            
//             int64_t* d_results;
//             cudaMalloc(&d_results, tableSize * sizeof(int64_t));
//             cudaMemset(d_results, 0, tableSize * sizeof(int64_t));
            
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
//             cudaMemcpy(resultVector.data(), d_results, tableSize * sizeof(int64_t), cudaMemcpyDeviceToHost);
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
//                   int64_t *d_results;
                  
//                   cudaMalloc(&d_leftCol, tableSize * sizeof(int));
//                   cudaMalloc(&d_rightCol, tableSize * sizeof(int));
//                   cudaMalloc(&d_results, tableSize * sizeof(int64_t));
                  
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
//                   cudaMemcpy(resultVector.data(), d_results, tableSize * sizeof(int64_t), cudaMemcpyDeviceToHost);
                  
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
//                   int64_t* d_results;
                  
//                   cudaMalloc(&d_leftStrings, tableSize * sizeof(char*));
//                   cudaMalloc(&d_rightStrings, tableSize * sizeof(char*));
//                   cudaMalloc(&d_results, tableSize * sizeof(int64_t));
                  
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
//                   cudaMemcpy(resultVector.data(), d_results, tableSize * sizeof(int64_t), cudaMemcpyDeviceToHost);
                  
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
// std::vector<int64_t> GPUManager::processComparisonExpr(
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
    
//     std::vector<int64_t> resultVector(resultSize, 0);
    
//     // Column-column comparison
//     if (expr->expr->type == hsql::kExprColumnRef && expr->expr2->type == hsql::kExprColumnRef) {
//         const char* leftColName = expr->expr->name;
//         const char* rightColName = expr->expr2->name;
        
//         int leftColIndex = findColumnIndex(leftTable, leftColName, expr->expr->table);
//         int rightColIndex = findColumnIndex(rightTable, rightColName, expr->expr2->table);
        
//         if (leftColIndex == -1 || rightColIndex == -1) {
//             // throw std::runtime_error("Column not found in comparison");
//             return  std::vector<int64_t> (resultSize, 1);
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
//             int64_t *d_results;
            
//             cudaMalloc(&d_leftCol, leftSize * sizeof(int));
//             cudaMalloc(&d_rightCol, rightSize * sizeof(int));
//             cudaMalloc(&d_results, resultSize * sizeof(int64_t));
            
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
//             cudaMemcpy(resultVector.data(), d_results, resultSize * sizeof(int64_t), cudaMemcpyDeviceToHost);
            
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
//             int64_t* d_results;
            
//             cudaMalloc(&d_leftStrings, leftSize * sizeof(char*));
//             cudaMalloc(&d_rightStrings, rightSize * sizeof(char*));
//             cudaMalloc(&d_results, resultSize * sizeof(int64_t));
            
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
//             cudaMemcpy(resultVector.data(), d_results, resultSize * sizeof(int64_t), cudaMemcpyDeviceToHost);
            
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

// std::vector<int64_t> GPUManager::processBinaryExpr(
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
//               int64_t *d_leftResults, *d_rightResults, *d_output;
//               cudaMalloc(&d_leftResults, resultSize * sizeof(int64_t));
//               cudaMalloc(&d_rightResults, resultSize * sizeof(int64_t));
//               cudaMalloc(&d_output, resultSize * sizeof(int64_t));
              
//               // Copy data to device
//               cudaMemcpy(d_leftResults, leftResults.data(), resultSize * sizeof(int64_t), cudaMemcpyHostToDevice);
//               cudaMemcpy(d_rightResults, rightResults.data(), resultSize * sizeof(int64_t), cudaMemcpyHostToDevice);
              
//               // Set up kernel execution parameters
//               int blockSize = 256;
//               int numBlocks = (resultSize + blockSize - 1) / blockSize;
              
//               // Execute kernel
//               int64_t isAnd = expr->opType == hsql::OperatorType::kOpAnd ? 1 : 0;
//               combineResults<<<numBlocks, blockSize>>>(d_leftResults, d_rightResults, d_output, resultSize, isAnd);
              
//               // Copy results back to host
//               std::vector<int64_t> resultVector(resultSize);
//               cudaMemcpy(resultVector.data(), d_output, resultSize * sizeof(int64_t), cudaMemcpyDeviceToHost);
              
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
//     return std::vector<int64_t>(resultSize, 1);  // 1 means true
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


