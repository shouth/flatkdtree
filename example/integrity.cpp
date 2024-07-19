#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <random>

#include <flatkdtree/flatkdtree.h>
#include <vector>

int main(int argc, const char *argv[])
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " count search [test]" << std::endl;
    return 1;
  }

  int count = std::atoi(argv[1]);
  int search = std::atoi(argv[2]);
  long test = argc >= 4 ? std::atoi(argv[3]) : 100;

  std::random_device gen;
  std::mt19937_64 engine{ gen() };
  std::uniform_real_distribution<double> dist{ 0, 1 };

  using Point = std::array<double, 2>;

  std::vector<Point> ps(count);
  std::vector<Point> qs(test);

  for (auto &p : ps) {
    p = { dist(gen), dist(gen) };
  }
  for (auto &q : qs) {
    q = { dist(gen), dist(gen) };
  }

  std::cout << "integrity check" << std::endl;

  std::vector<Point> ps2 = ps;
  std::vector<Point> points(search);
  std::vector<double> distances(search);

  auto start = std::chrono::high_resolution_clock::now();
  kdtree::construct(std::begin(ps), std::end(ps));
  for (std::size_t i = 0; i < test; ++i) {
    auto cmp = [&](const Point &lhs, const Point &rhs) {
      auto ldist = (lhs[0] - qs[i][0]) * (lhs[0] - qs[i][0]) + (lhs[1] - qs[i][1]) * (lhs[1] - qs[i][1]);
      auto rdist = (rhs[0] - qs[i][0]) * (rhs[0] - qs[i][0]) + (rhs[1] - qs[i][1]) * (rhs[1] - qs[i][1]);
      return ldist < rdist;
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

    auto result = std::mismatch(std::begin(ps2), std::begin(ps2) + search, std::begin(points));
    if (result.first != std::begin(ps2) + search) {
      std::size_t j = std::distance(std::begin(ps2), result.first);
      std::cerr << "Mismatch at " << i << ", " << j << std::endl;
      std::cerr << "    Expected: " << (*result.first)[0] << ", " << (*result.first)[1] << std::endl;
      std::cerr << "    Actual: " << (*result.second)[0] << ", " << (*result.second)[1] << std::endl;
      return 1;
    }
  }

  std::cout << "integrity check passed" << std::endl;
}
