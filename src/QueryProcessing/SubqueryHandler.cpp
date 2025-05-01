// #include "QueryProcessing/SubqueryHandler.hpp"
// #include "QueryProcessing/QueryExecutor.hpp"
// #include "Utilities/ErrorHandling.hpp"
// #include <hsql/util/sqlhelper.h>

// SubqueryHandler::SubqueryHandler(std::shared_ptr<QueryExecutor> executor)
//     : executor(executor) {}

// std::shared_ptr<Table> SubqueryHandler::handle(const hsql::SelectStatement *subquery)
// {
//     if (!subquery)
//     {
//         throw SemanticError("Null subquery provided");
//     }
//     return executor->execute(subquery);
// }

// std::shared_ptr<Table> SubqueryHandler::handleExists(const hsql::SelectStatement *subquery)
// {
//     auto result = handle(subquery);
//     // EXISTS returns true if any rows exist
//     return std::make_shared<Table>("exists_result",
//                                    std::vector<std::string>{"exists"},
//                                    std::vector<std::vector<std::string>>{
//                                        {result->getData().empty() ? "false" : "true"}});
// }

// std::shared_ptr<Table> SubqueryHandler::handleIn(const hsql::Expr *expr)
// {
//     if (!expr || expr->type != hsql::kExprOperator ||
//         !expr->exprList || expr->exprList->empty())
//     {
//         throw SemanticError("Invalid IN expression");
//     }

//     // For IN subqueries
//     if (expr->exprList->size() == 1 &&
//         expr->exprList->at(0)->type == hsql::kExprSelect)
//     {
//         return handle(expr->exprList->at(0)->select);
//     }

//     // For IN value lists (1,2,3)
//     std::vector<std::vector<std::string>> values;
//     for (const hsql::Expr *value : *expr->exprList)
//     {
//         if (value->type == hsql::kExprLiteralInt)
//         {
//             values.push_back({std::to_string(value->ival)});
//         }
//         else if (value->type == hsql::kExprLiteralString)
//         {
//             values.push_back({value->name});
//         }
//         else
//         {
//             throw SemanticError("Unsupported IN value type");
//         }
//     }

//     return std::make_shared<Table>("in_values",
//                                    std::vector<std::string>{"value"},
//                                    values);
// }