#ifndef SPECULA_PROF_HPP_
#define SPECULA_PROF_HPP_

#ifdef ENABLE_PROF
#include <chrono>
#include <cstdio>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <thread>
#include <typeinfo>

#define PROF_STRINGIFY_IMPL(x) #x
#define PROF_STRINGIFY(x) PROF_STRINGIFY_IMPL(x)
#define PROF_CONCAT4_IMPL(a, b, c, d) a##b##c##d
#define PROF_CONCAT4(a, b, c, d) PROF_CONCAT4_IMPL(a, b, c, d)

#define PROF_FE0(func, obj, ...)
#define PROF_FE1(func, obj, x, ...)                                            \
  func(obj, x) PROF_FE0(func, obj, __VA_ARGS__)
#define PROF_FE2(func, obj, x, ...)                                            \
  func(obj, x) PROF_FE1(func, obj, __VA_ARGS__)
#define PROF_FE3(func, obj, x, ...)                                            \
  func(obj, x) PROF_FE2(func, obj, __VA_ARGS__)
#define PROF_FE4(func, obj, x, ...)                                            \
  func(obj, x) PROF_FE3(func, obj, __VA_ARGS__)
#define PROF_FE5(func, obj, x, ...)                                            \
  func(obj, x) PROF_FE4(func, obj, __VA_ARGS__)
#define PROF_FE6(func, obj, x, ...)                                            \
  func(obj, x) PROF_FE5(func, obj, __VA_ARGS__)
#define PROF_FE7(func, obj, x, ...)                                            \
  func(obj, x) PROF_FE6(func, obj, __VA_ARGS__)
#define PROF_FE8(func, obj, x, ...)                                            \
  func(obj, x) PROF_FE7(func, obj, __VA_ARGS__)
#define PROF_FE10(func, obj, x, ...)                                           \
  func(obj, x) PROF_FE9(func, obj, __VA_ARGS__)
#define PROF_GET_FE(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, NAME, ...) NAME
#define PROF_FOR_EACH(action, obj, ...)                                        \
  PROF_GET_FE(__VA_ARGS__, PROF_FE10, PROF_FE9, PROF_FE8, PROF_FE7, PROF_FE6,  \
              PROF_FE5, PROF_FE4, PROF_FE3, PROF_FE2, PROF_FE1, PROF_FE0)      \
  (action, obj, __VA_ARGS__)

#define PROF_SCOPED_NAME                                                       \
  PROF_CONCAT4(__ProfilerScoped_, __LINE__, _, __COUNTER__)
#define PROF_SCOPED_NAME_STR PROF_STRINGIFY(PROF_SCOPED_NAME)
#define PROF_FUNCTION __PRETTY_FUNCTION__

#define PROF_SCOPED(...)                                                       \
  prof::ScopedProfiler PROF_SCOPED_NAME = prof::ScopedProfiler(__VA_ARGS__);
#define PROF_FUNC(...) PROF_SCOPED(PROF_FUNCTION, __VA_ARGS__)
#define PROF_FUNC_KEY(_0, val) , #val, val
#define PROF_FUNC_ARGS(...)                                                    \
  PROF_SCOPED(PROF_FUNCTION PROF_FOR_EACH(PROF_FUNC_KEY, obj, __VA_ARGS__));

#define PROF_BEGIN(...) prof::event_begin(__VA_ARGS__);
#define PROF_END(...) prof::event_end(__VA_ARGS__);
#define PROF_INST(...) prof::event_instant(__VA_ARGS__);
#define PROF_COUNT(...) prof::event_counter(__VA_ARGS__);

#define PROF_OBJ(type) prof::ObjectProfiler<type>
#define PROF_OBJ_CONSTRUCT(ptr) prof::event_object_construct(ptr);
#define PROF_OBJECT_DESTRUCT(ptr) prof::event_object_destroy(ptr);

#define PROF_KEY_VAL(obj, key) , #key, (obj)->key
#define PROF_SNAPSHOT(obj, ...)                                                \
  prof::event_object_snapshot(                                                 \
      obj PROF_FOR_EACH(PROF_KEY_VAL, obj, __VA_ARGS__));

// #ifdef ENABLE_FILE_STREAM
#define PROF_STREAM_FILE(file) prof::fs::open_stream_file(file);
#define PROF_CLOSE_STREAM() prof::fs::close_stream_file();
// #else
// #define PROF_STREAM_FILE(file)
// #define PROF_CLOSE_STREAM()
// #endif // ENABLE_FILE_STREAM
#else
#define PROF_SCOPED_NAME
#define PROF_SCOPED_NAME_STR
#define PROF_FUNCTION

#define PROF_SCOPED(...)
#define PROF_FUNC(...)
#define PROF_FUNC_KEY(_0, val)
#define PROF_FUNC_ARGS(...)

#define PROF_BEGIN(...)
#define PROF_END(...)
#define PROF_INST(...)
#define PROF_COUNT(...)

#define PROF_OBJ_CONSTRUCT(type, ptr)
#define PROF_OBJECT_DESTRUCT(type, ptr)

#define PROF_KEY_VAL(obj, key)
#define PROF_SNAPSHOT(type, obj, ...)
#define PROF_STREAM_FILE(file)
#define PROF_CLOSE_STREAM()
#endif // ENABLE_PROF

namespace prof {
#ifdef ENABLE_PROF
struct Event {
  enum EventType {
    BEGIN,
    END,
    INSTANCE,
    COUNTER,
    OBJECT_CONSTRUCT,
    OBJECT_DESTROY,
    OBJECT_SNAPSHOT
  };

  inline static char get_type_char(const EventType &type) {
    switch (type) {
    case BEGIN:
    default:
      return 'B';
    case END:
      return 'E';
    case INSTANCE:
      return 'I';
    case COUNTER:
      return 'C';
    case OBJECT_CONSTRUCT:
      return 'N';
    case OBJECT_DESTROY:
      return 'D';
    case OBJECT_SNAPSHOT:
      return 'O';
    }
  }

  EventType type;
  const std::string name;
  const std::string cat;
  const std::string args;
  const std::size_t tid;
  const std::size_t id;
};
extern std::hash<std::thread::id> thread_hasher;
extern std::hash<const void *> pointer_hasher;

// #ifdef ENABLE_FILE_STREAM
namespace fs {
extern FILE *file_stream;
inline void open_stream_file(const char *file) {
  file_stream = std::fopen(file, "w");
  std::fprintf(file_stream, "[");
}
inline void close_stream_file() {
  if (file_stream)
    std::fclose(file_stream);
}
inline void handle_event(const Event &event) {
  if (!file_stream)
    open_stream_file("profile.json");
  fmt::print(
      file_stream, "{{{}{}\"ph\":\"{}\",\"ts\":{},\"pid\":1,\"tid\":{}{}{}}},",
      (event.name.size() == 0) ? ""
                               : fmt::format("\"name\":\"{}\",", event.name),
      (event.cat.size() == 0) ? "" : fmt::format("\"cat\":\"{}\",", event.cat),
      Event::get_type_char(event.type),
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count(),
      event.tid,
      (event.id == 0) ? "" : fmt::format(",\"id\":\"0x{:X}\"", event.id),
      (event.args.size() == 0) ? ""
                               : fmt::format(",\"args\":{{{}}}", event.args));
}
} // namespace fs
// #endif // ENABLE_FILE_STREAM

template <typename T> inline std::string fmt_type(const T &v) {
  return fmt::format("{}", v);
}
template <> inline std::string fmt_type(const std::string &v) {
  return fmt::format("\"{}\"", v);
}
template <typename T> inline std::string fmt_type(const std::vector<T> &v) {
  std::string res = "[";
  for (auto &it : v) {
    res += fmt_type(it) + ',';
  }
  return res.substr(0, v.size() != 0 ? res.size() - 1 : res.size()) + "]";
}

inline std::string fmt_args() { return ""; }
template <typename T>
inline std::string fmt_args(const std::string &key, const T &v) {
  return fmt::format("\"{}\": {}", key, fmt_type(v));
}
template <typename T, typename... ARGS>
inline std::string fmt_args(const std::string &key, const T &v,
                            const ARGS &... args) {
  return fmt::format("{}, \"{}\": {}", fmt_args(args...), key, fmt_type(v));
}
inline std::pair<std::string, std::string> split_cat_args() {
  return std::make_pair("", "");
}
template <typename... ARGS>
inline typename std::enable_if<(sizeof...(ARGS)) % 2 == 0,
                               std::pair<std::string, std::string>>::type
split_cat_args(const std::string &cat, const ARGS &... args) {
  return std::make_pair(cat, fmt_args(args...));
}
template <typename... ARGS>
inline typename std::enable_if<(sizeof...(ARGS)) % 2 != 0,
                               std::pair<std::string, std::string>>::type
split_cat_args(const std::string &cat, const ARGS &... args) {
  return std::make_pair("", fmt_args(cat, args...));
}

inline void handle_event(const Event &event) {
  // #ifdef ENABLE_FILE_STREAM
  fs::handle_event(event);
  // #endif // ENABLE_FILE_STREAM
}

template <typename... ARGS>
inline void event_begin(const std::string &name, const ARGS &... args) {
  auto cat_args = split_cat_args(args...);
  Event event{Event::EventType::BEGIN,
              name,
              cat_args.first,
              cat_args.second,
              thread_hasher(std::this_thread::get_id()),
              0};
  handle_event(event);
}
template <typename... ARGS> inline void event_end(const ARGS &... args) {
  Event event{Event::EventType::END,
              "",
              "",
              fmt_args(args...),
              thread_hasher(std::this_thread::get_id()),
              0};
  handle_event(event);
}
template <typename... ARGS>
inline void event_instant(const std::string &name, const ARGS &... args) {
  auto cat_args = split_cat_args(args...);
  Event event{Event::EventType::INSTANCE,
              name,
              cat_args.first,
              cat_args.second,
              thread_hasher(std::this_thread::get_id()),
              0};
  handle_event(event);
}
template <typename... ARGS>
inline void event_counter(const std::string &name, const ARGS &... args) {
  Event event{Event::EventType::COUNTER,
              name,
              "",
              fmt_args(args...),
              thread_hasher(std::this_thread::get_id()),
              0};
  handle_event(event);
}
template <typename T> inline void event_object_construct(const T *obj) {
  std::string type_str = typeid(&obj).name();
  type_str = type_str.substr(type_str.find_first_not_of(
      "0123456789", type_str.find_first_of("0123456789")));
  Event event{Event::EventType::OBJECT_CONSTRUCT,
              type_str,
              "",
              "",
              thread_hasher(std::this_thread::get_id()),
              pointer_hasher(reinterpret_cast<const void *>(obj))};
  handle_event(event);
}
template <typename T> inline void event_object_destroy(const T *obj) {
  std::string type_str = typeid(&obj).name();
  type_str = type_str.substr(type_str.find_first_not_of(
      "0123456789", type_str.find_first_of("0123456789")));
  Event event{Event::EventType::OBJECT_DESTROY,
              type_str,
              "",
              "",
              thread_hasher(std::this_thread::get_id()),
              pointer_hasher(reinterpret_cast<const void *>(obj))};
  handle_event(event);
}
template <typename T, typename... ARGS>
inline void event_object_snapshot(const T *obj, const ARGS &... args) {
  std::string type_str = typeid(&obj).name();
  type_str = type_str.substr(type_str.find_first_not_of(
      "0123456789", type_str.find_first_of("0123456789")));
  Event event{Event::EventType::OBJECT_SNAPSHOT,
              type_str,
              "",
              fmt::format("\"snapshot\":{{{}}}", fmt_args(args...)),
              thread_hasher(std::this_thread::get_id()),
              pointer_hasher(reinterpret_cast<const void *>(obj))};
  handle_event(event);
}
struct ScopedProfiler {
  template <typename... ARGS>
  ScopedProfiler(const std::string &name, const ARGS &... args) {
    prof::event_begin(name, args...);
  }
  ~ScopedProfiler() { prof::event_end(); }
};
template <typename T> struct ObjectProfiler {
  ObjectProfiler(const T *ptr) : ptr(ptr) { prof::event_object_construct(ptr); }
  ~ObjectProfiler() { prof::event_object_destroy(ptr); }

private:
  const T *ptr;
};
#endif // ENABLE_PROF
} // namespace prof

#endif // SPECULA_PROF_HPP_