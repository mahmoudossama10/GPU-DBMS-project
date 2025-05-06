// #pragma once
// #include <vector>
// #include <string>

// namespace GPUAggregator {


//     // Result struct for batched aggregates
//     struct IntAggregates {
//         long long sum;
//         int count;
//         int min;
//         int max;
//         double avg; // convenience
//         IntAggregates() : sum(0), count(0), min(0), max(0), avg(0.0) {}
//     };

//     // Batched kernel: does all in one call
//     IntAggregates multiAggregateInt(const std::vector<int>& col);

//     int sumInt(const std::vector<int>& col);
//     int minInt(const std::vector<int>& col);
//     int maxInt(const std::vector<int>& col);
//     int countInt(const std::vector<int>& col);
//     double avgInt(const std::vector<int>& col);

//     std::vector<int> gpuArgsortInt(const std::vector<int>& col, bool ascending=true);
// }