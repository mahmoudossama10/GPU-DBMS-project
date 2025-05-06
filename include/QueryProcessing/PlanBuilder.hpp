#pragma once
#include <memory>
#include <hsql/SQLParser.h>
#include <duckdb.hpp>
#include <duckdb.h>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/execution/physical_plan_generator.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_any_join.hpp>
#include <duckdb/planner/operator/logical_cross_product.hpp>

#include <duckdb/planner/expression/bound_reference_expression.hpp>

#include <duckdb/planner/operator/logical_join.hpp>

#include <duckdb/planner/joinside.hpp>
#include <hsql/SQLParserResult.h>
#include <hsql/sql/Expr.h>
#include <hsql/util/sqlhelper.h>
#include "../DataHandling/Table.hpp"
#include "../DataHandling/StorageManager.hpp"
#include "GPU.hpp" // Include the GPU header
// Flag to control GPU acceleration
struct GPUSortColumn;
struct RowIndexValue;

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
    // GPUJoinPlan(std::vector<std::shared_ptr<Table>> tables,
    //             std::vector<std::string> table_names,
    //             const hsql::Expr *where_clause,
    //             std::shared_ptr<GPUManager> gpu_manager);

    GPUJoinPlan(std::unique_ptr<ExecutionPlan> leftTable,
                std::unique_ptr<ExecutionPlan> rightTable,
                std::string where,
                std::shared_ptr<GPUManager> gpu_manager);

    std::shared_ptr<Table> execute() override;

private:
    std::vector<std::shared_ptr<Table>> tables_;
    std::vector<std::string> table_names_;
    const hsql::Expr *where_clause_;
    std::unique_ptr<ExecutionPlan> leftTable_;
    std::unique_ptr<ExecutionPlan> rightTable_;
    std::string whereString;
    std::shared_ptr<GPUManager> gpu_manager_;
};

class GPUOrderByPlan : public ExecutionPlan
{
public:
    GPUOrderByPlan(std::shared_ptr<Table> input,
                   const std::vector<hsql::OrderDescription *> &order_exprs,
                   std::shared_ptr<GPUManager> gpu_manager);
    std::shared_ptr<Table> execute() override;

private:
    std::shared_ptr<Table> input_;
    std::vector<hsql::OrderDescription *> order_exprs_;
    std::shared_ptr<GPUManager> gpu_manager_;
};

class GPUAggregatorPlan : public ExecutionPlan
{
public:
    GPUAggregatorPlan(std::shared_ptr<Table> input, const std::vector<hsql::Expr *> &select_list, std::shared_ptr<GPUManager> gpu_manager);

    std::shared_ptr<Table> execute() override;

private:
    std::shared_ptr<Table> input_;
    std::vector<hsql::Expr *> select_list_;
    std::shared_ptr<GPUManager> gpu_manager_;
};
class GPUJoinPlanMultipleTable : public ExecutionPlan
{
public:
    // GPUJoinPlan(std::vector<std::shared_ptr<Table>> tables,
    //             std::vector<std::string> table_names,
    //             const hsql::Expr *where_clause,
    //             std::shared_ptr<GPUManager> gpu_manager);

    GPUJoinPlanMultipleTable(std::vector<std::unique_ptr<ExecutionPlan>> tables,
                             std::string where,
                             std::shared_ptr<GPUManager> gpu_manager);

    std::shared_ptr<Table> execute() override;

private:
    std::vector<std::shared_ptr<Table>> tablesData_;
    std::vector<std::string> table_names_;
    const hsql::Expr *where_clause_;
    std::vector<std::unique_ptr<ExecutionPlan>> tablesExecutionPlan_;
    std::string whereString;
    std::shared_ptr<GPUManager> gpu_manager_;
};

class PlanBuilder
{
public:
    explicit PlanBuilder(std::shared_ptr<StorageManager> storage, ExecutionMode mode = ExecutionMode::GPU);
    bool isSelectAll(const std::vector<hsql::Expr *> *selectList);
    bool selectListNeedsProjection(const std::vector<hsql::Expr *> &selectList);
    bool hasAggregates(const std::vector<hsql::Expr *> &select_list);

    // Set execution mode (CPU or GPU)

    // Main build method
    std::unique_ptr<ExecutionPlan> build(const hsql::SelectStatement *stmt, const std::string &query);

    std::unique_ptr<ExecutionPlan> convertDuckDBPlanToExecutionPlan(const hsql::SelectStatement *stmt,
                                                                    std::unique_ptr<duckdb::LogicalOperator> duckdb_plan, int cnt);

    std::shared_ptr<Table> buildOrderByPlan(
        std::shared_ptr<Table> input,
        const std::vector<hsql::OrderDescription *> &order_exprs);

    std::shared_ptr<Table> buildProjectPlan(
        std::shared_ptr<Table> input,
        const std::vector<hsql::Expr *> &select_list);

    std::unique_ptr<ExecutionPlan> buildCPUJoinPlan(
        std::vector<std::unique_ptr<ExecutionPlan>> inputs,
        std::string where);

    std::shared_ptr<Table> buildGPUOrderByPlan(
        std::shared_ptr<Table> input,
        const std::vector<hsql::OrderDescription *> &order_exprs);

    std::shared_ptr<Table> buildCPUAggregatePlan(
        std::shared_ptr<Table> input,
        const std::vector<hsql::Expr *> &select_list);

    std::shared_ptr<Table> buildGPUAggregatePlan(
        std::shared_ptr<Table> input,
        const std::vector<hsql::Expr *> &select_list);

    static std::shared_ptr<Table> output_join_table;
    static int joinPlansCount;

    static void setExecutionMode(ExecutionMode mode)
    {
        PlanBuilder::execution_mode_ = mode;
    }

private:
    std::shared_ptr<StorageManager> storage_;
    static ExecutionMode execution_mode_;
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
                                                   const std::string where);

    // Extract tables from complex table references
    std::vector<std::pair<std::string, std::string>> extractTableReferences(const hsql::TableRef *table);

    // Helper to process where clauses for GPU execution
    hsql::Expr *processWhereClause(const hsql::Expr *where);

    // Other plan builders

    std::unique_ptr<ExecutionPlan> buildAggregatePlan(
        std::unique_ptr<ExecutionPlan> input,
        const std::vector<hsql::Expr *> &select_list);

    std::unique_ptr<ExecutionPlan> buildGPUJoinPlan(
        std::unique_ptr<ExecutionPlan> leftTable,
        std::unique_ptr<ExecutionPlan> rightTable,
        std::string where);

    // Check if a table reference has a subquery
    bool hasSubqueryInTableRef(const hsql::TableRef *table);

    std::unique_ptr<ExecutionPlan> buildGPUJoinPlanMultipleTable(
        std::vector<std::unique_ptr<ExecutionPlan>> tables,
        std::string where);

    std::unique_ptr<ExecutionPlan> buildPassPlane(
        std::unique_ptr<ExecutionPlan> input);
};