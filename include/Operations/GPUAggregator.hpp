#pragma once
#include <vector>
#include <string>

namespace GPUAggregator {
    int sumInt(const std::vector<int>& col);
    int minInt(const std::vector<int>& col);
    int maxInt(const std::vector<int>& col);
    int countInt(const std::vector<int>& col);
    double avgInt(const std::vector<int>& col);

    std::vector<int> gpuArgsortInt(const std::vector<int>& col, bool ascending=true);
}