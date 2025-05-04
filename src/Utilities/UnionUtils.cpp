#include "Utilities/UnionUtils.hpp"
#include <algorithm>
#include <sstream>
#include <cstring>
#include <string>
#include <sstream>
#include <iomanip>

namespace UnionUtils
{
    std::string valueToString(unionV aggValue, ColumnType aggType)
    {
        switch (aggType)
        {
        case ColumnType::STRING:
            return *(aggValue.s);

        case ColumnType::INTEGER:
            return std::to_string(aggValue.i);

        case ColumnType::DOUBLE:
            return std::to_string(aggValue.d);

        case ColumnType::DATETIME:
        {
            dateTime *dt = aggValue.t;
            std::ostringstream oss;
            oss << std::setw(4) << std::setfill('0') << dt->year << "-"
                << std::setw(2) << std::setfill('0') << dt->month << "-"
                << std::setw(2) << std::setfill('0') << dt->day << " "
                << std::setw(2) << std::setfill('0') << static_cast<int>(dt->hour) << ":"
                << std::setw(2) << std::setfill('0') << static_cast<int>(dt->minute) << ":"
                << std::setw(2) << std::setfill('0') << static_cast<int>(dt->second);
            return oss.str();
        }

        default:
            return "UNKNOWN";
        }
    }

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