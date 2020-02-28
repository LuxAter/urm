#ifndef FMT_HPP_
#define FMT_HPP_

#include <cctype>
#include <cstring>
#include <stdexcept>
#include <string>

namespace fmt {
template <typename... Args>
std::basic_string<char> format(const std::basic_string<char> &fmt_spec,
                               const Args &... args) {
  char *buffer = nullptr;
  size_t buffer_size;
  buffer_size = std::snprintf(NULL, 0, fmt_spec.data(), args...) + 2;
  buffer = (char *)std::malloc(sizeof(char) * buffer_size);
  std::snprintf(buffer, buffer_size * sizeof(char), fmt_spec.data(), args...);
  std::basic_string<char> output(buffer);
  free(buffer);
  return output;
}
template <typename... Args>
std::basic_string<char> format(const char *&fmt_spec, const Args &... args) {
  char *buffer = nullptr;
  size_t buffer_size;
  buffer_size = std::snprintf(NULL, 0, fmt_spec, args...);
  buffer = (char *)std::malloc(sizeof(char) * buffer_size) + 2;
  std::snprintf(buffer, buffer_size * sizeof(char), fmt_spec, args...);
  std::basic_string<char> output(buffer);
  free(buffer);
  return output;
}
template <typename... Args>
void print(const std::basic_string<char> &fmt_spec, const Args &... args) {
  std::fprintf(stdout, fmt_spec.data(), args...);
}
template <typename... Args>
void print(const char *fmt_spec, const Args &... args) {
  std::fprintf(stdout, fmt_spec, args...);
}
} // namespace fmt

#endif // FMT_HPP_
