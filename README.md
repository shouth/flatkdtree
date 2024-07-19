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

### Adapt your own type to flatkdtree

```cpp
// your own type
struct MyPoint
{
  double x, y;
};

// specialize `access` and `dimension` for your own type
template <std::size_t I>
struct kdtree::trait::access<MyPoint, I>
{
  static double get(const MyPoint &p)
  {
    return I == 0 ? p.x : p.y;
  }
};

template <typename T>
struct kdtree::trait::dimension<MyPoint>
{
  static constexpr std::size_t value = 2;
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
