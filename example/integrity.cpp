#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <random>

#include <flatkdtree/flatkdtree.h>

int main(int argc, const char *argv[])
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " count search [test]" << std::endl;
    return 1;
  }

  auto count = std::atoi(argv[1]);
  auto search = std::atoi(argv[2]);
  auto test = argc >= 4 ? std::atoi(argv[3]) : 100;

  auto gen = std::random_device{};
  auto engine = std::mt19937{ gen() };
  auto dist = std::uniform_real_distribution<double>{ 0, 1e3 };

  using Point = std::array<double, 2>;

  auto ps = std::vector<Point>(count);
  auto qs = std::vector<Point>(test);

  for (auto &p : ps) {
    p = { dist(gen), dist(gen) };
  }
  for (auto &q : qs) {
    q = { dist(gen), dist(gen) };
  }

  std::cout << "integrity check" << std::endl;

  auto ps2 = ps;
  auto points = std::vector<Point>(search);
  auto distances = std::vector<double>(search);

  auto start = std::chrono::high_resolution_clock::now();
  kdtree::construct(std::begin(ps), std::end(ps));
  for (std::size_t i = 0; i < test; ++i) {
    auto cmp = [&](const auto &lhs, const auto &rhs) {
      auto policy = kdtree::default_point_policy<Point>{};
      return policy.distance(lhs, qs[i]) < policy.distance(rhs, qs[i]);
    };
    std::size_t found = kdtree::search_knn(std::begin(ps), std::end(ps), std::begin(points), std::begin(distances), search, qs[i]);
    std::sort(std::begin(points), std::end(points), cmp);

    if (found != search) {
      std::cerr << "Mismatch at " << i << std::endl;
      std::cerr << "    Expected: " << search << std::endl;
      std::cerr << "    Actual: " << found << std::endl;
      return 1;
    }

    std::partial_sort(std::begin(ps2), std::begin(ps2) + search, std::end(ps2), cmp);

    auto [p1, p2] = std::mismatch(std::begin(ps2), std::begin(ps2) + search, std::begin(points));
    if (p1 != std::begin(ps2) + search) {
      std::size_t j = std::distance(std::begin(ps2), p1);
      std::cerr << "Mismatch at " << i << ", " << j << std::endl;
      std::cerr << "    Expected: " << (*p1)[0] << ", " << (*p1)[1] << std::endl;
      std::cerr << "    Actual: " << (*p2)[0] << ", " << (*p2)[1] << std::endl;
      return 1;
    }
  }

  std::cout << "integrity check passed" << std::endl;
}
