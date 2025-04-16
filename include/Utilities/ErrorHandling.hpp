#pragma once
#include <stdexcept>
#include <string>

class QueryError : public std::runtime_error
{
public:
    explicit QueryError(const std::string &msg) : std::runtime_error(msg) {}
};

class SyntaxError : public QueryError
{
public:
    explicit SyntaxError(const std::string &msg) : QueryError("Syntax error: " + msg) {}
};

class SemanticError : public QueryError
{
public:
    explicit SemanticError(const std::string &msg) : QueryError("Semantic error: " + msg) {}
};

class IOError : public QueryError
{
public:
    explicit IOError(const std::string &msg) : QueryError("I/O error: " + msg) {}
};