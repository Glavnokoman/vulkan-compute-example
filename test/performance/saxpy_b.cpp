#include <benchmark/benchmark.h>

#include <example_filter.h>
#include <vulkan_helpers.hpp>

namespace {

   void BM_StringCreation(benchmark::State& state) {
     for (auto _ : state)
       std::string empty_string;
   }

   // Define another benchmark
   void BM_StringCopy(benchmark::State& state) {
     std::string x = "hello";
     for (auto _ : state)
       std::string copy(x);
   }
}

BENCHMARK(BM_StringCreation);
BENCHMARK(BM_StringCopy);

BENCHMARK_MAIN();
