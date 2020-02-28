#include <chrono>
#include <cstdlib>
#include <iostream>
#include <cmath>

// #define VML_FMT
// #define VML_OSTREAM
// #define VML_OPENACC
// #include "fmt.hpp"
// #include "vml.hpp"

#define NUM 100000

struct vec {
  float x, y, z;
};

int main(int argc, char *argv[]) {

  vec a[NUM];
  float v[NUM];
  float sum = 0;
  for (size_t i = 0; i < NUM; ++i) {
    a[i] =
        vec({rand() % 1000 * 1.0f, rand() % 1000 * 1.0f, rand() % 1000 * 1.0f});
  }

  auto start = std::chrono::high_resolution_clock::now();
#pragma acc parallel loop copy(a, v)
  for (size_t i = 0; i < NUM; ++i) {
    for (size_t j = 0; j < 32; ++j) {
      v[i] = std::pow(std::pow(a[i].x, 0.3) + std::pow(a[i].y, 0.6) +
                          std::pow(a[i].z, 1 / M_PI),
                      0.5);
    }
    sum += v[i];
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::cout << sum << ':' << v[NUM / 2] << "\n";
  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(stop -
                                                                     start)
                       .count() /
                   32.0
            << '\n';

  return 0;
}
