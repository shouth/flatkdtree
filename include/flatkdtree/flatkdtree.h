//   Copyright 2024 Shota Minami
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#ifndef FLATKDTREE_FLATKDTREE_H_
#define FLATKDTREE_FLATKDTREE_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace kdtree
{
template <typename T, typename Enabler = void>
struct default_point_policy;

template <typename Float, std::size_t Size>
struct default_point_policy<std::array<Float, Size>, std::enable_if_t<std::is_floating_point_v<Float>>>
{
  using point_type = std::array<Float, Size>;
  using element_type = Float;
  using distance_type = Float;

  static constexpr std::size_t dimension = Size;

  template <std::size_t Index>
  constexpr auto get(const point_type &point) const -> element_type
  {
    return point[Index];
  }

  constexpr auto distance(const point_type &lhs, const point_type &rhs) const -> distance_type
  {
    distance_type dist = 0;
    for (std::size_t i = 0; i < Size; ++i) {
      dist += (lhs[i] - rhs[i]) * (lhs[i] - rhs[i]);
    }
    return dist;
  }
};

namespace internal
{
  template <typename Iterator>
  using iter_value_type = typename std::iterator_traits<Iterator>::value_type;

  template <int Dimension = 0, typename RandomAccessIterator, typename Policy>
  auto construct_recursive(
    RandomAccessIterator first, RandomAccessIterator last, const Policy &policy) -> void
  {
    if (first == last) {
      return;
    }

    auto middle = first + std::distance(first, last) / 2;
    auto get = [&](const auto &point) { return policy.template get<Dimension>(point); };

    std::nth_element(first, middle, last, [&](const auto &lhs, const auto &rhs) {
      return get(lhs) < get(rhs);
    });

    static constexpr auto NextDimension = (Dimension + 1) % Policy::dimension;
    construct_recursive<NextDimension>(first, middle, policy);
    construct_recursive<NextDimension>(middle + 1, last, policy);
  }

  template <
    std::size_t Dimension = 0,
    typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3,
    typename Policy = default_point_policy<iter_value_type<RandomAccessIterator1>>>
  auto search_knn_recursive(
    RandomAccessIterator1 first, RandomAccessIterator1 last,
    RandomAccessIterator2 out_point, RandomAccessIterator3 out_distance,
    std::size_t k, std::size_t n, const internal::iter_value_type<RandomAccessIterator1> &query,
    const Policy &policy = { })
    -> std::size_t
  {
    if (first == last) {
      return n;
    }

    const auto middle = first + std::distance(first, last) / 2;
    const auto distance = policy.distance(query, *middle);

    if (n < k or distance < *out_distance) {
      std::size_t i, j;

      if (n == k) {
        --n;
        for (i = 0;; i = j) {
          std::size_t l = i * 2 + 1;
          std::size_t r = i * 2 + 2;
          if (r < k) {
            j = out_distance[l] > out_distance[r] ? l : r;
          } else if (l < k) {
            j = l;
          } else {
            break;
          }

          if (out_distance[j] < out_distance[n]) {
            break;
          }

          out_point[i] = out_point[j];
          out_distance[i] = out_distance[j];
        }

        out_point[i] = out_point[n];
        out_distance[i] = out_distance[n];
      }

      for (i = n; i > 0; i = j) {
        j = (i - 1) / 2;
        if (out_distance[j] > distance) {
          break;
        }

        out_point[i] = out_point[j];
        out_distance[i] = out_distance[j];
      }

      out_point[i] = *middle;
      out_distance[i] = distance;
      ++n;
    }

    const auto get = [&](const auto &point) {
      return policy.template get<Dimension>(point);
    };
    const auto search = [&](auto first, auto last) {
      constexpr auto NextDimension = (Dimension + 1) % Policy::dimension;
      return search_knn_recursive<NextDimension>(first, last, out_point, out_distance, k, n, query, policy);
    };

    if (get(query) < get(*middle)) {
      n = search(first, middle);

      if (get(query) + *out_distance > get(*middle)) {
        n = search(middle + 1, last);
      }
    } else {
      n = search(middle + 1, last);

      if (get(query) - *out_distance < get(*middle)) {
        n = search(first, middle);
      }
    }

    return n;
  }
}

template <
  typename RandomAccessIterator,
  typename Policy = default_point_policy<internal::iter_value_type<RandomAccessIterator>>>
auto construct(RandomAccessIterator first, RandomAccessIterator last, const Policy &policy = { })
{
  internal::construct_recursive(first, last, policy);
}

template <
  typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3,
  typename Policy = default_point_policy<internal::iter_value_type<RandomAccessIterator1>>>
auto search_knn(
  RandomAccessIterator1 first, RandomAccessIterator1 last,
  RandomAccessIterator2 out_point, RandomAccessIterator3 out_distance,
  std::size_t k, const internal::iter_value_type<RandomAccessIterator1> &query,
  const Policy &policy = { })
  -> std::size_t
{
  return internal::search_knn_recursive(first, last, out_point, out_distance, k, 0, query, policy);
}
}

#endif  // FLATKDTREE_FLATKDTREE_H_
