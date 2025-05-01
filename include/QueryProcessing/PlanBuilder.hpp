#pragma once
#include <memory>
#include <hsql/SQLParser.h>
#include <duckdb.hpp>
#include <duckdb.h>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/execution/physical_plan_generator.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include "../DataHandling/Table.hpp"
#include "../DataHandling/StorageManager.hpp"
#include "GPU.hpp" // Include the GPU header

// Flag to control GPU acceleration
enum class ExecutionMode
{
    CPU,
    GPU
};

class ExecutionPlan
{
public:
    virtual ~ExecutionPlan() = default;
    virtual std::shared_ptr<Table> execute() = 0;
};

// New GPU-specific execution plan for table scans that involve joins
class GPUJoinPlan : public ExecutionPlan
{
public:
    GPUJoinPlan(std::vector<std::shared_ptr<Table>> tables,
                std::vector<std::string> table_names,
                const hsql::Expr *where_clause,
                std::shared_ptr<GPUManager> gpu_manager);

    std::shared_ptr<Table> execute() override;

private:
    std::vector<std::shared_ptr<Table>> tables_;
    std::vector<std::string> table_names_;
    const hsql::Expr *where_clause_;
    std::shared_ptr<GPUManager> gpu_manager_;
};

class PlanBuilder
{
public:
    explicit PlanBuilder(std::shared_ptr<StorageManager> storage, ExecutionMode mode = ExecutionMode::CPU);

    // Set execution mode (CPU or GPU)
    void setExecutionMode(ExecutionMode mode);

    // Main build method
    std::unique_ptr<ExecutionPlan> build(const hsql::SelectStatement *stmt, const std::string &query);

    std::unique_ptr<ExecutionPlan> convertDuckDBPlanToExecutionPlan(
        std::unique_ptr<duckdb::LogicalOperator> duckdb_plan);

private:
    std::shared_ptr<StorageManager> storage_;
    ExecutionMode execution_mode_;
    std::shared_ptr<GPUManager> gpu_manager_;

    // Subquery processing
    hsql::Expr *processWhereWithSubqueries(const hsql::Expr *expr);
    hsql::Expr *processSubqueryExpression(const hsql::Expr *expr);

    // Table scan plans (CPU and GPU)
    std::unique_ptr<ExecutionPlan> buildScanPlan(const hsql::TableRef *table);
    std::unique_ptr<ExecutionPlan> buildGPUScanPlan(const hsql::TableRef *table, const hsql::Expr *where);
    std::shared_ptr<Table> processSubqueryInFrom(const hsql::TableRef *table);

    // Filter plans
    std::unique_ptr<ExecutionPlan> buildFilterPlan(std::unique_ptr<ExecutionPlan> input,
                                                   const hsql::Expr *where);

    // Extract tables from complex table references
    std::vector<std::pair<std::string, std::string>> extractTableReferences(const hsql::TableRef *table);

    // Helper to process where clauses for GPU execution
    hsql::Expr *processWhereClause(const hsql::Expr *where);

    // Other plan builders
    std::unique_ptr<ExecutionPlan> buildProjectPlan(
        std::unique_ptr<ExecutionPlan> input,
        const std::vector<hsql::Expr *> &select_list);

    std::unique_ptr<ExecutionPlan> buildAggregatePlan(
        std::unique_ptr<ExecutionPlan> input,
        const std::vector<hsql::Expr *> &select_list);

    std::unique_ptr<ExecutionPlan> buildOrderByPlan(
        std::unique_ptr<ExecutionPlan> input,
        const std::vector<hsql::OrderDescription *> &order_exprs);

    // Check if a table reference has a subquery
    bool hasSubqueryInTableRef(const hsql::TableRef *table);
};