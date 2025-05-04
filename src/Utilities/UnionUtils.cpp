#include "Utilities/UnionUtils.hpp"
#include <algorithm>
#include <sstream>
#include <cstring>

namespace UnionUtils
{

    unionV copyUnionValue(const unionV &value, ColumnType type)
    {
        unionV copy;

        switch (type)
        {
        case ColumnType::STRING:
            copy.s = (value.s != nullptr) ? new std::string(*value.s) : nullptr;
            break;
        case ColumnType::INTEGER:
            copy.i = value.i;
            break;
        case ColumnType::DOUBLE:
            copy.d = value.d;
            break;
        case ColumnType::DATETIME:
            if (value.t != nullptr)
            {
                copy.t = new dateTime;
                *copy.t = *value.t;
            }
            else
            {
                copy.t = nullptr;
            }
            break;
        }

        return copy;
    }

}