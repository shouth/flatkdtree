#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include <flatkdtree/flatkdtree.h>

class benchmark
{
  std::string _name;
  std::vector<double> _times;
  std::chrono::high_resolution_clock::time_point _start;

  public:
  benchmark(const std::string &name)
      : _name(name)
  {
  }

  auto start() -> void
  {
    _start = std::chrono::high_resolution_clock::now();
  }

  auto stop() -> void
  {
    auto end = std::chrono::high_resolution_clock::now();
    auto ellapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - _start).count() / 1e6;
    _times.push_back(ellapsed);
  }

  auto report() -> void
  {
    auto sum = std::accumulate(std::begin(_times), std::end(_times), 0.0);
    auto mean = sum / _times.size();
    auto variance = std::accumulate(std::begin(_times), std::end(_times), 0.0, [&](auto acc, auto time) {
      return acc + (time - mean) * (time - mean);
    }) / _times.size();
    auto stddev = std::sqrt(variance);
    auto min = *std::min_element(std::begin(_times), std::end(_times));
    auto max = *std::max_element(std::begin(_times), std::end(_times));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << _name << std::endl;
    std::cout << "    mean: " << mean << " ± " << stddev << "ms" << std::endl;
    std::cout << "    min: " << min << "ms" << std::endl;
    std::cout << "    max: " << max << "ms" << std::endl;
  }
};

int main(int argc, const char *argv[])
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " count search [iteration]" << std::endl;
    return 1;
  }

  auto count = std::atoi(argv[1]);
  auto search = std::atoi(argv[2]);
  auto iteration = argc >= 4 ? std::atoi(argv[3]) : 100;

  auto gen = std::random_device{};
  auto engine = std::mt19937{ gen() };
  auto dist = std::uniform_real_distribution<double>{ 0, 1e3 };

  using Point = std::array<double, 2>;

  auto ps = std::vector<Point>(count);
  auto qs = std::vector<Point>(iteration);

  for (auto &p : ps) {
    p = { dist(gen), dist(gen) };
  }
  for (auto &q : qs) {
    q = { dist(gen), dist(gen) };
  }

  {
    auto b = benchmark{ "construct" };
    for (std::size_t i = 0; i < iteration; ++i) {
      std::shuffle(std::begin(ps), std::end(ps), engine);
      b.start();
      kdtree::construct(std::begin(ps), std::end(ps));
      b.stop();
    }
    b.report();
  }

  {
    auto points = std::vector<Point>(search);
    auto distances = std::vector<double>(search);
    auto b = benchmark{ "search" };
    for (std::size_t i = 0; i < iteration; ++i) {
      b.start();
      kdtree::search_knn(std::begin(ps), std::end(ps), std::begin(points), std::begin(distances), search, qs[i]);
      b.stop();
    }
    b.report();
  }
}
