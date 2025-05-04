#pragma once

#include <string>
#include <vector>
#include <cctype>

struct dateTime
{
    unsigned short year;
    unsigned short month;
    unsigned short day;
    unsigned char hour;
    unsigned char minute;
    unsigned char second;
};

enum class ColumnType
{
    STRING,
    INTEGER,
    DOUBLE,
    DATETIME
};

union unionV
{
    std::string *s;
    int64_t i;
    double d;
    dateTime *t;
};

namespace UnionUtils
{

    unionV copyUnionValue(const unionV &value, ColumnType type);

}