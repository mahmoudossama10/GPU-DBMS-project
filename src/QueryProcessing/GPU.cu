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



// Helper method to recursively process batches across multiple tables
std::shared_ptr<Table> GPUManager::executeMultipleTableJoin(
    const std::vector<std::shared_ptr<Table>>& tables,
    const hsql::Expr* joinConditions) {
    
    // Base case: processed all tables in this batch combination
        // Process this specific batch combination
        auto startProcess = std::chrono::high_resolution_clock::now();

        auto batchResults = processBatch(tables, joinConditions);
        std::vector<std::vector<unionV>> resultData;

        auto endProcess = std::chrono::high_resolution_clock::now();
std::chrono::duration<double, std::milli> processTime = endProcess - startProcess;
std::cout << "processBatch time: " << processTime.count() << " ms" << std::endl;

auto startFilter = std::chrono::high_resolution_clock::now();
        // Extract matching rows from the batch
        std::vector<std::vector<int>> selectedCombinations;
        int64_t totalBatchSize = 1;
        for (const auto& table : tables) {
            totalBatchSize *= table->getSize();
        }

        // Find all combinations that matched (where batchResults[i] == 1)
        for (int64_t i = 0; i < totalBatchSize; i++) {
            if (batchResults[i] == 1) {
                std::vector<int> combination;
                int64_t index = i;
                if(selectedCombinations.size()>=995){
                    int x=0;
                }
                for (int t = 0; t < tables.size(); t++) {
                    int64_t tableSize = tables[t]->getSize();
                    int64_t mod = index % tableSize;
                    combination.insert(combination.end(), mod);
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

// Process a specific batch combination across all tables
std::vector<int64_t> GPUManager::processBatch(
    const std::vector<std::shared_ptr<Table>>& tables,
    const hsql::Expr* conditions) {
    
    // Calculate batch size
    int batchSize = 1;
    for (const auto& table : tables) {
        batchSize *= table->getSize();
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
            auto leftResults = processBatch(tables, conditions->expr);
            auto rightResults = processBatch(tables, conditions->expr2);
            
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
            results = evaluateConditionOnBatch(tables, conditions);
        }
    }
    
    return results;
}

// Improved version of evaluateConditionOnBatch for column-major data and integer columns
std::vector<int64_t> GPUManager::evaluateConditionOnBatch(
    const std::vector<std::shared_ptr<Table>>& tables,
    const hsql::Expr* condition) {
    
    // Calculate total batch size (cartesian product of all tables)
    int batchSize = 1;
    for (const auto& table : tables) {
        batchSize *= table->getSize();
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
        int leftBatchSize = leftTable->getSize();
        int rightBatchSize = rightTable->getSize();
        
        // Get integer column data directly from the Table class
        // (using the cached int columns from the Table implementation)
        const auto& leftColumnData = leftTable->getData().at(leftColName);
        const auto& rightColumnData = rightTable->getData().at(rightColName);
        
        

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
        for (const auto& table : tables) {
            tableSizes.push_back(table->getSize());
        }
        
        cudaMalloc(&d_tableSizes, tableSizes.size() * sizeof(int));
        
        // Async memory transfers
        cudaMemcpyAsync(d_leftCol, leftColumnData.data(), leftBatchSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_rightCol, rightColumnData.data(), rightBatchSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
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
    std::vector<int64_t> results;
    results.reserve(totalCombinations); // Only allocates memory, no initialization

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
    // hsql::printExpression(condition, 5);
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
    cudaFree(d_prefixSum);
    cudaFree(d_result);
    free(h_result);
    // Create and return the joined table
    return std::make_shared<Table>("joined_result", headers, columnData, columnTypes);
}





//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////                    ///////////////////////////////////
///////////////////////////////       SORTING      ///////////////////////////////////
///////////////////////////////                    ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

// CUDA error checking macro
#define CUDA_CHECK(call)                                                  \
    do                                                                    \
    {                                                                     \
        cudaError_t error = call;                                         \
        if (error != cudaSuccess)                                         \
        {                                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
            std::cerr << ": " << cudaGetErrorString(error) << std::endl;  \
            throw std::runtime_error("CUDA error");                       \
        }                                                                 \
    } while (0)






// CUDA kernel for preparing row indices
__global__ void prepareRowIndices(const int *int_data, const double *double_data,
    const char *string_data, size_t *string_offsets,
    RowIndexValue *row_indices, size_t num_rows, size_t num_cols,
    GPUSortColumn *sort_columns, int num_sort_columns,
    int primary_sort_column_index)
{
size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
if (row_idx >= num_rows)
return;

// Set the row index
row_indices[row_idx].row_index = row_idx;
}

// CUDA comparison function for bitonic sort - STABLE version

__device__ bool compareRowIndices(const RowIndexValue &a, const RowIndexValue &b,
    const GPUSortColumn *sort_columns, int num_sort_cols, size_t num_cols, size_t original_num_rows,
    const int *int_data, const double *double_data,
    const char *string_data, size_t *string_offsets)
{
if (a.row_index >= original_num_rows && b.row_index < original_num_rows)
{
return false; // a > b
}
if (a.row_index < original_num_rows && b.row_index >= original_num_rows)
{
return true; // a < b
}
if (a.row_index >= original_num_rows && b.row_index >= original_num_rows)
{
return a.row_index < b.row_index;
}

for (int i = 0; i < num_sort_cols; i++)
{
const GPUSortColumn &col = sort_columns[i];
int cmp = 0;
switch (col.type)
{
case 1: // INTEGER
{
int val_a = int_data[a.row_index * num_cols + col.column_index];
int val_b = int_data[b.row_index * num_cols + col.column_index];
cmp = (val_a < val_b) ? -1 : (val_a > val_b ? 1 : 0);
}
break;
case 2: // DOUBLE
{
double val_a = double_data[a.row_index * num_cols + col.column_index];
double val_b = double_data[b.row_index * num_cols + col.column_index];
cmp = (val_a < val_b) ? -1 : (val_a > val_b ? 1 : 0);
}
break;
case 3: // DATETIME
{
double val_a = double_data[a.row_index * num_cols + col.column_index];
double val_b = double_data[b.row_index * num_cols + col.column_index];
cmp = (val_a < val_b) ? -1 : (val_a > val_b ? 1 : 0);
}
break;
case 0: // STRING
{
size_t offset_a = string_offsets[a.row_index * num_cols + col.column_index];
size_t offset_b = string_offsets[b.row_index * num_cols + col.column_index];
int j = 0;
while (true)
{
char char_a = string_data[offset_a + j];
char char_b = string_data[offset_b + j];
if (char_a != char_b)
{
cmp = char_a - char_b;
break;
}
if (char_a == '\0')
{
cmp = 0;
break;
}
j++;
}
}
break;
default:
break;
}

if (cmp != 0)
{
return col.is_ascending ? (cmp < 0) : (cmp > 0);
}
}

return a.row_index < b.row_index; // Stable sort
}

// CUDA kernel for bitonic sort step
__global__ void bitonicSortStep(RowIndexValue *row_indices, size_t n,
  unsigned int j, unsigned int k,
  const GPUSortColumn *sort_columns, int num_sort_cols, size_t num_cols, size_t original_num_rows,
  const int *int_data, const double *double_data,
  const char *string_data, size_t *string_offsets)
{
unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
if (i >= n)
return;

unsigned int ij = i ^ j;
if (ij > i)
{
bool swap = false;
if ((i & k) == 0)
{
swap = !compareRowIndices(row_indices[i], row_indices[ij],
        sort_columns, num_sort_cols, num_cols, original_num_rows,
        int_data, double_data, string_data, string_offsets);
}
else
{
swap = compareRowIndices(row_indices[i], row_indices[ij],
       sort_columns, num_sort_cols, num_cols, original_num_rows,
       int_data, double_data, string_data, string_offsets);
}
if (swap)
{
RowIndexValue temp = row_indices[i];
row_indices[i] = row_indices[ij];
row_indices[ij] = temp;
}
}
}

   


std::shared_ptr<Table> GPUManager::executeOrderBy(
    std::shared_ptr<Table> table,
    const std::vector<hsql::OrderDescription *> &order_exprs_){

        // Execute input plan

    if (!table || table->getData().empty())
        return table;

    // Parse ORDER BY
    std::vector<SortColumn> sort_cols = parseOrderBy(*table, order_exprs_);
    if (sort_cols.empty())
        return table;

    // Get table dimensions
    const auto &data_map = table->getData();
    const auto &headers = table->getHeaders();
    size_t num_rows = data_map.begin()->second.size();
    size_t num_cols = headers.size();
    size_t pow2_size = 1;
    while (pow2_size < num_rows)
    {
        pow2_size *= 2;
    }

    // Prepare data arrays
    std::vector<int> int_data(pow2_size * num_cols, 0);
    std::vector<double> double_data(pow2_size * num_cols, 0.0);
    std::vector<char> string_data;
    std::vector<size_t> string_offsets(pow2_size * num_cols, 0); // Flat array for all columns
    size_t string_data_offset = 0;

    // Populate data in headers order
    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx)
    {
        for (size_t col_idx = 0; col_idx < num_cols; ++col_idx)
        {
            const std::string &col_name = headers[col_idx];
            ColumnType col_type = table->getColumnType(col_name);
            const unionV &val = data_map.at(col_name)[row_idx];
            size_t flat_idx = row_idx * num_cols + col_idx;

            switch (col_type)
            {
            case ColumnType::INTEGER:
                int_data[flat_idx] = val.i;
                break;
            case ColumnType::DOUBLE:
                double_data[flat_idx] = val.d;
                break;
            case ColumnType::DATETIME:
                double_data[flat_idx] = val.d; // Handle DATETIME as DOUBLE
                break;
            case ColumnType::STRING:
            {
                std::string str = *val.s;
                string_offsets[flat_idx] = string_data_offset;
                for (char c : str)
                {
                    string_data.push_back(c);
                }
                string_data.push_back('\0');
                string_data_offset += str.length() + 1;
                break;
            }
            default:
                throw SemanticError("Unsupported column type in GPU ORDER BY");
            }
        }
    }

    // Pad string offsets for remaining rows
    for (size_t row_idx = num_rows; row_idx < pow2_size; ++row_idx)
    {
        for (size_t col_idx = 0; col_idx < num_cols; ++col_idx)
        {
            size_t flat_idx = row_idx * num_cols + col_idx;
            string_offsets[flat_idx] = string_data_offset;
        }
    }

    // Convert SortColumn to GPU-compatible struct
    std::vector<GPUSortColumn> gpu_sort_cols;
    for (const auto &col : sort_cols)
    {
        gpu_sort_cols.push_back({col.column_index, col.is_ascending, static_cast<int>(col.type)});
    }

    // Allocate GPU memory
    int *d_int_data = nullptr;
    double *d_double_data = nullptr;
    char *d_string_data = nullptr;
    size_t *d_string_offsets = nullptr;
    GPUSortColumn *d_sort_cols = nullptr;
    RowIndexValue *d_row_indices = nullptr;

    try
    {
        CUDA_CHECK(cudaMalloc(&d_int_data, int_data.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_double_data, double_data.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_string_data, string_data.size() * sizeof(char)));
        CUDA_CHECK(cudaMalloc(&d_string_offsets, string_offsets.size() * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_sort_cols, gpu_sort_cols.size() * sizeof(GPUSortColumn)));
        CUDA_CHECK(cudaMalloc(&d_row_indices, pow2_size * sizeof(RowIndexValue)));

        CUDA_CHECK(cudaMemcpy(d_int_data, int_data.data(), int_data.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_double_data, double_data.data(), double_data.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_string_data, string_data.data(), string_data.size() * sizeof(char), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_string_offsets, string_offsets.data(), string_offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sort_cols, gpu_sort_cols.data(), gpu_sort_cols.size() * sizeof(GPUSortColumn), cudaMemcpyHostToDevice));

        // Initialize row indices on host
        std::vector<RowIndexValue> row_indices(pow2_size);
        for (size_t i = 0; i < pow2_size; ++i)
        {
            row_indices[i].row_index = i;
        }
        CUDA_CHECK(cudaMemcpy(d_row_indices, row_indices.data(), pow2_size * sizeof(RowIndexValue), cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocksPerGrid = (pow2_size + threadsPerBlock - 1) / threadsPerBlock;

        // Perform bitonic sort
        for (unsigned int k = 2; k <= pow2_size; k <<= 1)
        {
            for (unsigned int j = k >> 1; j > 0; j >>= 1)
            {
                bitonicSortStep<<<blocksPerGrid, threadsPerBlock>>>(
                    d_row_indices, pow2_size, j, k, d_sort_cols, gpu_sort_cols.size(), num_cols, num_rows,
                    d_int_data, d_double_data, d_string_data, d_string_offsets);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        // Copy sorted indices back
        std::vector<RowIndexValue> sorted_row_indices(pow2_size);
        CUDA_CHECK(cudaMemcpy(sorted_row_indices.data(), d_row_indices, pow2_size * sizeof(RowIndexValue), cudaMemcpyDeviceToHost));

        // Create sorted table
        std::unordered_map<std::string, std::vector<unionV>> sorted_data_map;
        for (const auto &header : headers)
        {
            sorted_data_map[header].reserve(num_rows);
        }
        for (size_t i = 0; i < num_rows; ++i)
        {
            size_t orig_row_idx = sorted_row_indices[i].row_index;
            for (const auto &header : headers)
            {
                sorted_data_map[header].push_back(data_map.at(header)[orig_row_idx]);
            }
        }

        // Free GPU memory
        CUDA_CHECK(cudaFree(d_int_data));
        CUDA_CHECK(cudaFree(d_double_data));
        CUDA_CHECK(cudaFree(d_string_data));
        CUDA_CHECK(cudaFree(d_string_offsets));
        CUDA_CHECK(cudaFree(d_sort_cols));
        CUDA_CHECK(cudaFree(d_row_indices));

        return std::make_shared<Table>(
            table->getName() + "_ordered",
            headers,
            std::move(sorted_data_map),
            table->getColumnTypes());
    }
    catch (const std::exception &e)
    {
        // Clean up
        if (d_int_data)
            cudaFree(d_int_data);
        if (d_double_data)
            cudaFree(d_double_data);
        if (d_string_data)
            cudaFree(d_string_data);
        if (d_string_offsets)
            cudaFree(d_string_offsets);
        if (d_sort_cols)
            cudaFree(d_sort_cols);
        if (d_row_indices)
            cudaFree(d_row_indices);
        throw; // Propagate error
    }
    }









    std::vector<GPUManager::SortColumn> GPUManager::parseOrderBy(const Table &table, const std::vector<hsql::OrderDescription *> &order_exprs_){

        std::vector<GPUManager::SortColumn> sort_cols;
        const auto &headers = table.getHeaders();
    
        for (const auto *order_desc : order_exprs_)
        {
            if (order_desc->type != hsql::kOrderAsc && order_desc->type != hsql::kOrderDesc)
                throw std::runtime_error("Unsupported ORDER BY type");
    
            const hsql::Expr *expr = order_desc->expr;
            if (!expr || expr->type != hsql::kExprColumnRef)
                throw std::runtime_error("Only column references are supported in ORDER BY");
    
            std::string col_name = expr->name;
            size_t col_idx = table.getColumnIndex(col_name);
            if (col_idx >= headers.size() || headers[col_idx] != col_name)
            {
                // Verify index matches header
                for (size_t i = 0; i < headers.size(); ++i)
                {
                    if (headers[i] == col_name)
                    {
                        col_idx = i;
                        break;
                    }
                }
            }
    
            ColumnType col_type = table.getColumnType(col_name);
            sort_cols.push_back({col_idx, order_desc->type == hsql::kOrderAsc, col_type});
        }
    
        return sort_cols;
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