#pragma once
#include "../DataHandling/Table.hpp"
#include "../QueryProcessing/PlanBuilder.hpp"
#include <memory>
#include <vector>
#include <string>

class JoinPlan : public ExecutionPlan
{
public:
    JoinPlan(std::unique_ptr<ExecutionPlan> left,
             std::unique_ptr<ExecutionPlan> right,
             const std::string &left_alias = "",
             const std::string &right_alias = "");

    std::shared_ptr<Table> execute() override;

private:
    std::unique_ptr<ExecutionPlan> left_;
    std::unique_ptr<ExecutionPlan> right_;
    std::string left_alias_;
    std::string right_alias_;

    // Helper methods
    std::vector<std::string> combineHeaders(const Table &left,
                                            const Table &right) const;
    std::vector<std::vector<std::string>> computeProduct(
        const Table &left,
        const Table &right) const;
    std::string prefixHeader(const std::string &header,
                             const std::string &alias) const;
};