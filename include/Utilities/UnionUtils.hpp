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

struct TheInteger
{
    int64_t value;
    bool is_null;
};

struct TheDouble
{
    double value;
    bool is_null = false;
};

union unionV
{
    std::string *s;
    TheInteger *i;
    TheDouble *d;
    dateTime *t;
};

namespace UnionUtils
{

    unionV copyUnionValue(const unionV &value, ColumnType type);
    std::string valueToString(unionV aggValue, ColumnType aggType);
}