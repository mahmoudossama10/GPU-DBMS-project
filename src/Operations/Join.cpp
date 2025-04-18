#include "../../include/Operations/Join.hpp"
#include "../../include/Utilities/ErrorHandling.hpp"
#include <algorithm>

JoinPlan::JoinPlan(std::unique_ptr<ExecutionPlan> left,
                   std::unique_ptr<ExecutionPlan> right,
                   const std::string &left_alias,
                   const std::string &right_alias)
    : left_(std::move(left)),
      right_(std::move(right)),
      left_alias_(left_alias),
      right_alias_(right_alias) {}

std::shared_ptr<Table> JoinPlan::execute()
{
    auto left_table = left_->execute();
    auto right_table = right_->execute();

    // Combine metadata
    auto headers = combineHeaders(*left_table, *right_table);
    auto data = computeProduct(*left_table, *right_table);

    return std::make_shared<Table>(
        "JoinedTable",
        headers,
        data);
}

std::vector<std::string> JoinPlan::combineHeaders(
    const Table &left,
    const Table &right) const
{
    std::vector<std::string> headers;

    // Process left headers
    for (const auto &h : left.getHeaders())
    {
        headers.push_back(prefixHeader(h, left_alias_));
    }

    // Process right headers
    for (const auto &h : right.getHeaders())
    {
        headers.push_back(prefixHeader(h, right_alias_));
    }

    return headers;
}

std::vector<std::vector<std::string>> JoinPlan::computeProduct(
    const Table &left,
    const Table &right) const
{
    std::vector<std::vector<std::string>> result;

    // Cartesian product calculation
    for (const auto &l_row : left.getData())
    {
        for (const auto &r_row : right.getData())
        {
            auto combined = l_row;
            combined.insert(combined.end(), r_row.begin(), r_row.end());
            result.push_back(combined);
        }
    }

    return result;
}

std::string JoinPlan::prefixHeader(const std::string &header,
                                   const std::string &alias) const
{
    if (!alias.empty())
    {
        return alias + "." + header;
    }
    return header;
}