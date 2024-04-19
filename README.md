# flatkdtree

Yet another K-D tree library that uses a simple array. It supports C++11 and later.

## Example Usage

```cpp
int main()
{
  // point type. `std::array` of `float`, `double` or `long double` are accepted by default.
  using Point = std::array<float, 2>;
  
  // create an array of points
  std::vector<Point> points = /* ... */;
  
  // construct kd-tree on an array
  kdtree::construct(points.begin(), points.end());
  
  // search k-nn
  Point query = /* ... */;
  int k = 10;
  std::vector<Point> result_points(k);
  std::vector<float> result_distances(k);
  kdtree::search_knn(
    points.begin(), points.end(), result_points.begin(), result_distances.begin(), k, query);
}
```

## Installation

Copy [include/flatkdtree/flatkdtree.h](https://github.com/shouth/flatkdtree/tree/main/include/flatkdtree) to your project and include it.

If you are using Cmake, you can use `ExternalProject`.

## More Examples

### Adapting your own types by specializing `kdtree::default_point_policy`

```cpp
// your own type
struct MyPoint
{
  double x, y;
};

// specialize `kdtree::default_point_policy`
template <>
struct kdtree::default_point_policy<MyPoint>
{
  using point_type = MyPoint;
  using distance_type = double;

  static constexpr std::size_t dimension = 2;

  template <std::size_t Index>
  auto element_compare(const point_type &p, const point_type &q) const -> bool
  {
    if constexpr (Index == 0) {
      return p.x < q.x;
    } else {
      return p.y < q.y;
    }
  }

  template <std::size_t Index>
  auto element_distance(const point_type &p, const point_type &q) const -> distance_type
  {
    if constexpr (Index == 0) {
      return (p.x - q.x) * (p.x - q.x);
    } else {
      return (p.y - q.y) * (p.y - q.y);
    }
  }

  auto distance(const point_type &p, const point_type &q) const -> distance_type
  {
    return (p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y);
  }
};

int main()
{
  // create an array of points
  std::vector<MyPoint> points = /* ... */;
  
  // construct kd-tree on an array
  kdtree::construct(points.begin(), points.end());
  
  // search k-nn
  MyPoint query = /* ... */;
  int k = 10;
  std::vector<MyPoint> result_points(k);
  std::vector<double> result_distances(k);
  kdtree::search_knn(
    points.begin(), points.end(), result_points.begin(), result_distances.begin(), k, query);
}
```

### Adapting your own types by creating your own `PointPolicy`

```cpp
// your own type
struct MyPoint
{
  double x, y;
};

// create your own `PointPolicy`
struct MyPointPolicy
{
  using point_type = MyPoint;
  using distance_type = double;

  static constexpr std::size_t dimension = 2;

  template <std::size_t Index>
  auto element_compare(const point_type &p, const point_type &q) const -> bool
  {
    if constexpr (Index == 0) {
      return p.x < q.x;
    } else {
      return p.y < q.y;
    }
  }

  template <std::size_t Index>
  auto element_distance(const point_type &p, const point_type &q) const -> distance_type
  {
    if constexpr (Index == 0) {
      return (p.x - q.x) * (p.x - q.x);
    } else {
      return (p.y - q.y) * (p.y - q.y);
    }
  }

  auto distance(const point_type &p, const point_type &q) const -> distance_type
  {
    return (p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y);
  }
};

int main()
{
  // create an array of points
  std::vector<MyPoint> points = /* ... */;
  
  // construct kd-tree on an array
  // pass your own `PointPolicy` as last argument
  kdtree::construct(points.begin(), points.end(), MyPointPolicy{});
  
  // search k-nn
  MyPoint query = /* ... */;
  int k = 10;
  std::vector<MyPoint> result_points(k);
  std::vector<double> result_distances(k);
  // pass your own `PointPolicy` as last argument
  kdtree::search_knn(
    points.begin(), points.end(), result_points.begin(), result_distances.begin(), k, query, MyPointPolicy{});
}
```
