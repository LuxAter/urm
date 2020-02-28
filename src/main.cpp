#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

// #define VML_FMT
// #define VML_OSTREAM
// #define VML_OPENACC
// #include "fmt.hpp"
#include "vml.hpp"

#define NUM 100000

int main(int argc, char *argv[]) {

  vml::fvec3 a(1, 2, 3), b(4);
  std::cout << vml::fmt(a) << ":" << vml::fmt(b) << "\n";
  std::cout << vml::fmt(a.zyx) << ":" << vml::fmt(a.zzz) << vml::fmt(a.xxyy) << "\n";
  std::cout << vml::fmt(vml::sin(a.yyxz)) << ":" << vml::fmt(vml::sin(b)) << "\n";
  std::cout << vml::fmt(a + 2) << ":" << vml::fmt(a + b) << "\n";
  std::cout << vml::fmt(a - 2) << ":" << vml::fmt(a - b) << "\n";
  std::cout << vml::fmt(a * 2) << ":" << vml::fmt(a * b) << "\n";
  std::cout << vml::fmt(a / 2) << ":" << vml::fmt(a / b) << "\n";
  a += (b / 3);
  std::cout << vml::fmt(a) << ":" << vml::fmt(a.yyxz) << "\n";

  return 0;
}
