#ifndef VML_HPP_
#define VML_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#define VML_FUNC(NAME)                                                         \
  template <typename... Args>                                                  \
  inline auto NAME(Args &&... args)                                            \
      ->decltype(::vml::detail::NAME(                                          \
          ::vml::detail::decay(std::forward<Args>(args))...)) {                \
    return ::vml::detail::NAME(                                                \
        ::vml::detail::decay(std::forward<Args>(args))...);                    \
  }

namespace vml {
template <typename T, size_t N> struct vector;
template <typename T, size_t M, size_t N> struct matrix;
namespace detail {
  struct nothing {};

  template <size_t Begin, size_t End, class Func>
  inline constexpr typename std::enable_if<Begin == End, void>::type
  static_for(Func &&) {}
  template <size_t Begin, size_t End, class Func>
  inline constexpr typename std::enable_if<Begin != End, void>::type
  static_for(Func &&f) {
    f(Begin);
    static_for<Begin + 1, End>(std::forward<Func>(f));
  }

  template <typename T, size_t N, class Func>
  inline constexpr vector<T, N> static_constructor(Func &&f) {
    vector<T, N> tmp;
    return static_constructor<0, T, N>(tmp, std::forward<Func>(f));
  }
  template <size_t I, typename T, size_t N, class Func>
  inline constexpr typename std::enable_if<I != N, vector<T, N>>::type
  static_constructor(vector<T, N> &v, Func &&f) {
    v[I] = f(I);
    return static_constructor<I + 1, T, N>(v, std::forward<Func>(f));
  }
  template <size_t I, typename T, size_t N, class Func>
  inline constexpr typename std::enable_if<I == N, vector<T, N>>::type
  static_constructor(vector<T, N> &v, Func &&) {
    return v;
  }

  template <typename T, size_t M, size_t N, class Func>
  inline constexpr matrix<T, M, N> static_constructor(Func &&f) {
    matrix<T, M, N> tmp;
    return static_constructor<0, T, M, N>(tmp, f);
  }
  template <size_t I, typename T, size_t M, size_t N, class Func>
  inline constexpr typename std::enable_if<I != M, matrix<T, M, N>>::type
  static_constructor(matrix<T, M, N> &v, Func &&f) {
    v[I] = f(I);
    return static_constructor<I + 1, T, N>(v, std::forward<Func>(f));
  }
  template <size_t I, typename T, size_t M, size_t N, class Func>
  inline constexpr typename std::enable_if<I == M, matrix<T, M, N>>::type
  static_constructor(matrix<T, M, N> &v, Func &&) {
    return v;
  }

  template <typename T>
  inline constexpr auto decay(T &&t) -> decltype(t.decay()) {
    return t.decay();
  }
  template <typename T>
  inline constexpr typename std::enable_if<
      std::is_arithmetic<typename std::remove_reference<T>::type>::value,
      T>::type
  decay(T &&t) {
    return t;
  }

  template <typename T, size_t N, size_t M, unsigned int MASK> struct swizzle {
    typedef T scalar_type;
    typedef vector<T, M> vector_type;
    scalar_type data[N];

    scalar_type &operator[](size_t i) {
      return data[(MASK >> (8 * (M - i - 1))) & 0xFF];
    }
    const scalar_type &operator[](size_t i) const {
      return data[(MASK >> (8 * (M - i - 1))) & 0xFF];
    }

    vector_type decay() const {
      return ::vml::detail::static_constructor<T, M>(
          [&](size_t i) { return (*this)[i]; });
    }
    operator vector_type() const { return decay(); }
    operator vector_type() { return decay(); }
    operator typename std::conditional<M == 1, scalar_type,
                                       ::vml::detail::nothing>::type() const {
      return data[0];
    }
    swizzle &operator=(const vector_type &vec) {
      scalar_type tmp[M];
      for (size_t i = 0; i < M; ++i)
        tmp[i] = vec[i];
      for (size_t i = 0; i < M; ++i)
        *this[i] = tmp[i];
      return *this;
    }
  };

  template <typename T, size_t N> struct vector_base;

  template <typename T> struct vector_base<T, 1> {
    typedef T scalar_type;
    union {
      scalar_type data[1];
      swizzle<T, 1, 1, 0x00> x, r;
      swizzle<T, 1, 2, 0x0000> xx, rr;
      swizzle<T, 1, 3, 0x000000> xxx, rrr;
      swizzle<T, 1, 4, 0x00000000> xxxx, rrrr;
    };
  };

  template <typename T> struct vector_base<T, 2> {
    typedef T scalar_type;
    union {
      scalar_type data[2];
      swizzle<T, 2, 1, 0x00> x, r;
      swizzle<T, 2, 1, 0x01> y, g;

      swizzle<T, 2, 2, 0x0000> xx, rr;
      swizzle<T, 2, 2, 0x0001> xy, rg;
      swizzle<T, 2, 2, 0x0100> yx, gr;
      swizzle<T, 2, 2, 0x0101> yy, gg;

      swizzle<T, 2, 3, 0x000000> xxx, rrr;
      swizzle<T, 2, 3, 0x000001> xxy, rrg;
      swizzle<T, 2, 3, 0x000100> xyx, rgr;
      swizzle<T, 2, 3, 0x000101> xyy, rgg;
      swizzle<T, 2, 3, 0x010000> yxx, grr;
      swizzle<T, 2, 3, 0x010001> yxy, grg;
      swizzle<T, 2, 3, 0x010100> yyx, ggr;
      swizzle<T, 2, 3, 0x010101> yyy, ggg;

      swizzle<T, 2, 4, 0x00000000> xxxx, rrrr;
      swizzle<T, 2, 4, 0x00000001> xxxy, rrrg;
      swizzle<T, 2, 4, 0x00000100> xxyx, rrgr;
      swizzle<T, 2, 4, 0x00000101> xxyy, rrgg;
      swizzle<T, 2, 4, 0x00010000> xyxx, rgrr;
      swizzle<T, 2, 4, 0x00010001> xyxy, rgrg;
      swizzle<T, 2, 4, 0x00010100> xyyx, rggr;
      swizzle<T, 2, 4, 0x00010101> xyyy, rggg;
      swizzle<T, 2, 4, 0x01000000> yxxx, grrr;
      swizzle<T, 2, 4, 0x01000001> yxxy, grrg;
      swizzle<T, 2, 4, 0x01000100> yxyx, grgr;
      swizzle<T, 2, 4, 0x01000101> yxyy, grgg;
      swizzle<T, 2, 4, 0x01010000> yyxx, ggrr;
      swizzle<T, 2, 4, 0x01010001> yyxy, ggrg;
      swizzle<T, 2, 4, 0x01010100> yyyx, gggr;
      swizzle<T, 2, 4, 0x01010101> yyyy, gggg;
    };
  };

  template <typename T> struct vector_base<T, 3> {
    typedef T scalar_type;
    union {
      scalar_type data[3];
      swizzle<T, 3, 1, 0x00> x, r;
      swizzle<T, 3, 1, 0x01> y, g;
      swizzle<T, 3, 1, 0x02> z, b;

      swizzle<T, 3, 2, 0x0000> xx, rr;
      swizzle<T, 3, 2, 0x0001> xy, rg;
      swizzle<T, 3, 2, 0x0002> xz, rb;
      swizzle<T, 3, 2, 0x0100> yx, gr;
      swizzle<T, 3, 2, 0x0101> yy, gg;
      swizzle<T, 3, 2, 0x0102> yz, gb;
      swizzle<T, 3, 2, 0x0200> zx, br;
      swizzle<T, 3, 2, 0x0201> zy, bg;
      swizzle<T, 3, 2, 0x0202> zz, bb;

      swizzle<T, 3, 3, 0x000000> xxx, rrr;
      swizzle<T, 3, 3, 0x000001> xxy, rrg;
      swizzle<T, 3, 3, 0x000002> xxz, rrb;
      swizzle<T, 3, 3, 0x000100> xyx, rgr;
      swizzle<T, 3, 3, 0x000101> xyy, rgg;
      swizzle<T, 3, 3, 0x000102> xyz, rgb;
      swizzle<T, 3, 3, 0x000200> xzx, rbr;
      swizzle<T, 3, 3, 0x000201> xzy, rbg;
      swizzle<T, 3, 3, 0x000202> xzz, rbb;
      swizzle<T, 3, 3, 0x010000> yxx, grr;
      swizzle<T, 3, 3, 0x010001> yxy, grg;
      swizzle<T, 3, 3, 0x010002> yxz, grb;
      swizzle<T, 3, 3, 0x010100> yyx, ggr;
      swizzle<T, 3, 3, 0x010101> yyy, ggg;
      swizzle<T, 3, 3, 0x010102> yyz, ggb;
      swizzle<T, 3, 3, 0x010200> yzx, gbr;
      swizzle<T, 3, 3, 0x010201> yzy, gbg;
      swizzle<T, 3, 3, 0x010202> yzz, gbb;
      swizzle<T, 3, 3, 0x020000> zxx, brr;
      swizzle<T, 3, 3, 0x020001> zxy, brg;
      swizzle<T, 3, 3, 0x020002> zxz, brb;
      swizzle<T, 3, 3, 0x020100> zyx, bgr;
      swizzle<T, 3, 3, 0x020101> zyy, bgg;
      swizzle<T, 3, 3, 0x020102> zyz, bgb;
      swizzle<T, 3, 3, 0x020200> zzx, bbr;
      swizzle<T, 3, 3, 0x020201> zzy, bbg;
      swizzle<T, 3, 3, 0x020202> zzz, bbb;

      swizzle<T, 3, 4, 0x00000000> xxxx, rrrr;
      swizzle<T, 3, 4, 0x00000001> xxxy, rrrg;
      swizzle<T, 3, 4, 0x00000002> xxxz, rrrb;
      swizzle<T, 3, 4, 0x00000100> xxyx, rrgr;
      swizzle<T, 3, 4, 0x00000101> xxyy, rrgg;
      swizzle<T, 3, 4, 0x00000102> xxyz, rrgb;
      swizzle<T, 3, 4, 0x00000200> xxzx, rrbr;
      swizzle<T, 3, 4, 0x00000201> xxzy, rrbg;
      swizzle<T, 3, 4, 0x00000202> xxzz, rrbb;
      swizzle<T, 3, 4, 0x00010000> xyxx, rgrr;
      swizzle<T, 3, 4, 0x00010001> xyxy, rgrg;
      swizzle<T, 3, 4, 0x00010002> xyxz, rgrb;
      swizzle<T, 3, 4, 0x00010100> xyyx, rggr;
      swizzle<T, 3, 4, 0x00010101> xyyy, rggg;
      swizzle<T, 3, 4, 0x00010102> xyyz, rggb;
      swizzle<T, 3, 4, 0x00010200> xyzx, rgbr;
      swizzle<T, 3, 4, 0x00010201> xyzy, rgbg;
      swizzle<T, 3, 4, 0x00010202> xyzz, rgbb;
      swizzle<T, 3, 4, 0x00020000> xzxx, rbrr;
      swizzle<T, 3, 4, 0x00020001> xzxy, rbrg;
      swizzle<T, 3, 4, 0x00020002> xzxz, rbrb;
      swizzle<T, 3, 4, 0x00020100> xzyx, rbgr;
      swizzle<T, 3, 4, 0x00020101> xzyy, rbgg;
      swizzle<T, 3, 4, 0x00020102> xzyz, rbgb;
      swizzle<T, 3, 4, 0x00020200> xzzx, rbbr;
      swizzle<T, 3, 4, 0x00020201> xzzy, rbbg;
      swizzle<T, 3, 4, 0x00020202> xzzz, rbbb;
      swizzle<T, 3, 4, 0x01000000> yxxx, grrr;
      swizzle<T, 3, 4, 0x01000001> yxxy, grrg;
      swizzle<T, 3, 4, 0x01000002> yxxz, grrb;
      swizzle<T, 3, 4, 0x01000100> yxyx, grgr;
      swizzle<T, 3, 4, 0x01000101> yxyy, grgg;
      swizzle<T, 3, 4, 0x01000102> yxyz, grgb;
      swizzle<T, 3, 4, 0x01000200> yxzx, grbr;
      swizzle<T, 3, 4, 0x01000201> yxzy, grbg;
      swizzle<T, 3, 4, 0x01000202> yxzz, grbb;
      swizzle<T, 3, 4, 0x01010000> yyxx, ggrr;
      swizzle<T, 3, 4, 0x01010001> yyxy, ggrg;
      swizzle<T, 3, 4, 0x01010002> yyxz, ggrb;
      swizzle<T, 3, 4, 0x01010100> yyyx, gggr;
      swizzle<T, 3, 4, 0x01010101> yyyy, gggg;
      swizzle<T, 3, 4, 0x01010102> yyyz, gggb;
      swizzle<T, 3, 4, 0x01010200> yyzx, ggbr;
      swizzle<T, 3, 4, 0x01010201> yyzy, ggbg;
      swizzle<T, 3, 4, 0x01010202> yyzz, ggbb;
      swizzle<T, 3, 4, 0x01020000> yzxx, gbrr;
      swizzle<T, 3, 4, 0x01020001> yzxy, gbrg;
      swizzle<T, 3, 4, 0x01020002> yzxz, gbrb;
      swizzle<T, 3, 4, 0x01020100> yzyx, gbgr;
      swizzle<T, 3, 4, 0x01020101> yzyy, gbgg;
      swizzle<T, 3, 4, 0x01020102> yzyz, gbgb;
      swizzle<T, 3, 4, 0x01020200> yzzx, gbbr;
      swizzle<T, 3, 4, 0x01020201> yzzy, gbbg;
      swizzle<T, 3, 4, 0x01020202> yzzz, gbbb;
      swizzle<T, 3, 4, 0x02000000> zxxx, brrr;
      swizzle<T, 3, 4, 0x02000001> zxxy, brrg;
      swizzle<T, 3, 4, 0x02000002> zxxz, brrb;
      swizzle<T, 3, 4, 0x02000100> zxyx, brgr;
      swizzle<T, 3, 4, 0x02000101> zxyy, brgg;
      swizzle<T, 3, 4, 0x02000102> zxyz, brgb;
      swizzle<T, 3, 4, 0x02000200> zxzx, brbr;
      swizzle<T, 3, 4, 0x02000201> zxzy, brbg;
      swizzle<T, 3, 4, 0x02000202> zxzz, brbb;
      swizzle<T, 3, 4, 0x02010000> zyxx, bgrr;
      swizzle<T, 3, 4, 0x02010001> zyxy, bgrg;
      swizzle<T, 3, 4, 0x02010002> zyxz, bgrb;
      swizzle<T, 3, 4, 0x02010100> zyyx, bggr;
      swizzle<T, 3, 4, 0x02010101> zyyy, bggg;
      swizzle<T, 3, 4, 0x02010102> zyyz, bggb;
      swizzle<T, 3, 4, 0x02010200> zyzx, bgbr;
      swizzle<T, 3, 4, 0x02010201> zyzy, bgbg;
      swizzle<T, 3, 4, 0x02010202> zyzz, bgbb;
      swizzle<T, 3, 4, 0x02020000> zzxx, bbrr;
      swizzle<T, 3, 4, 0x02020001> zzxy, bbrg;
      swizzle<T, 3, 4, 0x02020002> zzxz, bbrb;
      swizzle<T, 3, 4, 0x02020100> zzyx, bbgr;
      swizzle<T, 3, 4, 0x02020101> zzyy, bbgg;
      swizzle<T, 3, 4, 0x02020102> zzyz, bbgb;
      swizzle<T, 3, 4, 0x02020200> zzzx, bbbr;
      swizzle<T, 3, 4, 0x02020201> zzzy, bbbg;
      swizzle<T, 3, 4, 0x02020202> zzzz, bbbb;
    };
  };

  template <typename T> struct vector_base<T, 4> {
    typedef T scalar_type;
    union {
      scalar_type data[4];
      swizzle<T, 4, 1, 0x00> x, r;
      swizzle<T, 4, 1, 0x01> y, g;
      swizzle<T, 4, 1, 0x02> z, b;
      swizzle<T, 4, 1, 0x03> w, a;

      swizzle<T, 4, 2, 0x0000> xx, rr;
      swizzle<T, 4, 2, 0x0001> xy, rg;
      swizzle<T, 4, 2, 0x0002> xz, rb;
      swizzle<T, 4, 2, 0x0003> xw, ra;
      swizzle<T, 4, 2, 0x0100> yx, gr;
      swizzle<T, 4, 2, 0x0101> yy, gg;
      swizzle<T, 4, 2, 0x0102> yz, gb;
      swizzle<T, 4, 2, 0x0103> yw, ga;
      swizzle<T, 4, 2, 0x0200> zx, br;
      swizzle<T, 4, 2, 0x0201> zy, bg;
      swizzle<T, 4, 2, 0x0202> zz, bb;
      swizzle<T, 4, 2, 0x0203> zw, ba;
      swizzle<T, 4, 2, 0x0300> wx, ar;
      swizzle<T, 4, 2, 0x0301> wy, ag;
      swizzle<T, 4, 2, 0x0302> wz, ab;
      swizzle<T, 4, 2, 0x0303> ww, aa;

      swizzle<T, 4, 3, 0x000000> xxx, rrr;
      swizzle<T, 4, 3, 0x000001> xxy, rrg;
      swizzle<T, 4, 3, 0x000002> xxz, rrb;
      swizzle<T, 4, 3, 0x000003> xxw, rra;
      swizzle<T, 4, 3, 0x000100> xyx, rgr;
      swizzle<T, 4, 3, 0x000101> xyy, rgg;
      swizzle<T, 4, 3, 0x000102> xyz, rgb;
      swizzle<T, 4, 3, 0x000103> xyw, rga;
      swizzle<T, 4, 3, 0x000200> xzx, rbr;
      swizzle<T, 4, 3, 0x000201> xzy, rbg;
      swizzle<T, 4, 3, 0x000202> xzz, rbb;
      swizzle<T, 4, 3, 0x000203> xzw, rba;
      swizzle<T, 4, 3, 0x000300> xwx, rar;
      swizzle<T, 4, 3, 0x000301> xwy, rag;
      swizzle<T, 4, 3, 0x000302> xwz, rab;
      swizzle<T, 4, 3, 0x000303> xww, raa;
      swizzle<T, 4, 3, 0x010000> yxx, grr;
      swizzle<T, 4, 3, 0x010001> yxy, grg;
      swizzle<T, 4, 3, 0x010002> yxz, grb;
      swizzle<T, 4, 3, 0x010003> yxw, gra;
      swizzle<T, 4, 3, 0x010100> yyx, ggr;
      swizzle<T, 4, 3, 0x010101> yyy, ggg;
      swizzle<T, 4, 3, 0x010102> yyz, ggb;
      swizzle<T, 4, 3, 0x010103> yyw, gga;
      swizzle<T, 4, 3, 0x010200> yzx, gbr;
      swizzle<T, 4, 3, 0x010201> yzy, gbg;
      swizzle<T, 4, 3, 0x010202> yzz, gbb;
      swizzle<T, 4, 3, 0x010203> yzw, gba;
      swizzle<T, 4, 3, 0x010300> ywx, gar;
      swizzle<T, 4, 3, 0x010301> ywy, gag;
      swizzle<T, 4, 3, 0x010302> ywz, gab;
      swizzle<T, 4, 3, 0x010303> yww, gaa;
      swizzle<T, 4, 3, 0x020000> zxx, brr;
      swizzle<T, 4, 3, 0x020001> zxy, brg;
      swizzle<T, 4, 3, 0x020002> zxz, brb;
      swizzle<T, 4, 3, 0x020003> zxw, bra;
      swizzle<T, 4, 3, 0x020100> zyx, bgr;
      swizzle<T, 4, 3, 0x020101> zyy, bgg;
      swizzle<T, 4, 3, 0x020102> zyz, bgb;
      swizzle<T, 4, 3, 0x020103> zyw, bga;
      swizzle<T, 4, 3, 0x020200> zzx, bbr;
      swizzle<T, 4, 3, 0x020201> zzy, bbg;
      swizzle<T, 4, 3, 0x020202> zzz, bbb;
      swizzle<T, 4, 3, 0x020203> zzw, bba;
      swizzle<T, 4, 3, 0x020300> zwx, bar;
      swizzle<T, 4, 3, 0x020301> zwy, bag;
      swizzle<T, 4, 3, 0x020302> zwz, bab;
      swizzle<T, 4, 3, 0x020303> zww, baa;
      swizzle<T, 4, 3, 0x030000> wxx, arr;
      swizzle<T, 4, 3, 0x030001> wxy, arg;
      swizzle<T, 4, 3, 0x030002> wxz, arb;
      swizzle<T, 4, 3, 0x030003> wxw, ara;
      swizzle<T, 4, 3, 0x030100> wyx, agr;
      swizzle<T, 4, 3, 0x030101> wyy, agg;
      swizzle<T, 4, 3, 0x030102> wyz, agb;
      swizzle<T, 4, 3, 0x030103> wyw, aga;
      swizzle<T, 4, 3, 0x030200> wzx, abr;
      swizzle<T, 4, 3, 0x030201> wzy, abg;
      swizzle<T, 4, 3, 0x030202> wzz, abb;
      swizzle<T, 4, 3, 0x030203> wzw, aba;
      swizzle<T, 4, 3, 0x030300> wwx, aar;
      swizzle<T, 4, 3, 0x030301> wwy, aag;
      swizzle<T, 4, 3, 0x030302> wwz, aab;
      swizzle<T, 4, 3, 0x030303> www, aaa;

      swizzle<T, 4, 4, 0x00000000> xxxx, rrrr;
      swizzle<T, 4, 4, 0x00000001> xxxy, rrrg;
      swizzle<T, 4, 4, 0x00000002> xxxz, rrrb;
      swizzle<T, 4, 4, 0x00000003> xxxw, rrra;
      swizzle<T, 4, 4, 0x00000100> xxyx, rrgr;
      swizzle<T, 4, 4, 0x00000101> xxyy, rrgg;
      swizzle<T, 4, 4, 0x00000102> xxyz, rrgb;
      swizzle<T, 4, 4, 0x00000103> xxyw, rrga;
      swizzle<T, 4, 4, 0x00000200> xxzx, rrbr;
      swizzle<T, 4, 4, 0x00000201> xxzy, rrbg;
      swizzle<T, 4, 4, 0x00000202> xxzz, rrbb;
      swizzle<T, 4, 4, 0x00000203> xxzw, rrba;
      swizzle<T, 4, 4, 0x00000300> xxwx, rrar;
      swizzle<T, 4, 4, 0x00000301> xxwy, rrag;
      swizzle<T, 4, 4, 0x00000302> xxwz, rrab;
      swizzle<T, 4, 4, 0x00000303> xxww, rraa;
      swizzle<T, 4, 4, 0x00010000> xyxx, rgrr;
      swizzle<T, 4, 4, 0x00010001> xyxy, rgrg;
      swizzle<T, 4, 4, 0x00010002> xyxz, rgrb;
      swizzle<T, 4, 4, 0x00010003> xyxw, rgra;
      swizzle<T, 4, 4, 0x00010100> xyyx, rggr;
      swizzle<T, 4, 4, 0x00010101> xyyy, rggg;
      swizzle<T, 4, 4, 0x00010102> xyyz, rggb;
      swizzle<T, 4, 4, 0x00010103> xyyw, rgga;
      swizzle<T, 4, 4, 0x00010200> xyzx, rgbr;
      swizzle<T, 4, 4, 0x00010201> xyzy, rgbg;
      swizzle<T, 4, 4, 0x00010202> xyzz, rgbb;
      swizzle<T, 4, 4, 0x00010203> xyzw, rgba;
      swizzle<T, 4, 4, 0x00010300> xywx, rgar;
      swizzle<T, 4, 4, 0x00010301> xywy, rgag;
      swizzle<T, 4, 4, 0x00010302> xywz, rgab;
      swizzle<T, 4, 4, 0x00010303> xyww, rgaa;
      swizzle<T, 4, 4, 0x00020000> xzxx, rbrr;
      swizzle<T, 4, 4, 0x00020001> xzxy, rbrg;
      swizzle<T, 4, 4, 0x00020002> xzxz, rbrb;
      swizzle<T, 4, 4, 0x00020003> xzxw, rbra;
      swizzle<T, 4, 4, 0x00020100> xzyx, rbgr;
      swizzle<T, 4, 4, 0x00020101> xzyy, rbgg;
      swizzle<T, 4, 4, 0x00020102> xzyz, rbgb;
      swizzle<T, 4, 4, 0x00020103> xzyw, rbga;
      swizzle<T, 4, 4, 0x00020200> xzzx, rbbr;
      swizzle<T, 4, 4, 0x00020201> xzzy, rbbg;
      swizzle<T, 4, 4, 0x00020202> xzzz, rbbb;
      swizzle<T, 4, 4, 0x00020203> xzzw, rbba;
      swizzle<T, 4, 4, 0x00020300> xzwx, rbar;
      swizzle<T, 4, 4, 0x00020301> xzwy, rbag;
      swizzle<T, 4, 4, 0x00020302> xzwz, rbab;
      swizzle<T, 4, 4, 0x00020303> xzww, rbaa;
      swizzle<T, 4, 4, 0x00030000> xwxx, rarr;
      swizzle<T, 4, 4, 0x00030001> xwxy, rarg;
      swizzle<T, 4, 4, 0x00030002> xwxz, rarb;
      swizzle<T, 4, 4, 0x00030003> xwxw, rara;
      swizzle<T, 4, 4, 0x00030100> xwyx, ragr;
      swizzle<T, 4, 4, 0x00030101> xwyy, ragg;
      swizzle<T, 4, 4, 0x00030102> xwyz, ragb;
      swizzle<T, 4, 4, 0x00030103> xwyw, raga;
      swizzle<T, 4, 4, 0x00030200> xwzx, rabr;
      swizzle<T, 4, 4, 0x00030201> xwzy, rabg;
      swizzle<T, 4, 4, 0x00030202> xwzz, rabb;
      swizzle<T, 4, 4, 0x00030203> xwzw, raba;
      swizzle<T, 4, 4, 0x00030300> xwwx, raar;
      swizzle<T, 4, 4, 0x00030301> xwwy, raag;
      swizzle<T, 4, 4, 0x00030302> xwwz, raab;
      swizzle<T, 4, 4, 0x00030303> xwww, raaa;
      swizzle<T, 4, 4, 0x01000000> yxxx, grrr;
      swizzle<T, 4, 4, 0x01000001> yxxy, grrg;
      swizzle<T, 4, 4, 0x01000002> yxxz, grrb;
      swizzle<T, 4, 4, 0x01000003> yxxw, grra;
      swizzle<T, 4, 4, 0x01000100> yxyx, grgr;
      swizzle<T, 4, 4, 0x01000101> yxyy, grgg;
      swizzle<T, 4, 4, 0x01000102> yxyz, grgb;
      swizzle<T, 4, 4, 0x01000103> yxyw, grga;
      swizzle<T, 4, 4, 0x01000200> yxzx, grbr;
      swizzle<T, 4, 4, 0x01000201> yxzy, grbg;
      swizzle<T, 4, 4, 0x01000202> yxzz, grbb;
      swizzle<T, 4, 4, 0x01000203> yxzw, grba;
      swizzle<T, 4, 4, 0x01000300> yxwx, grar;
      swizzle<T, 4, 4, 0x01000301> yxwy, grag;
      swizzle<T, 4, 4, 0x01000302> yxwz, grab;
      swizzle<T, 4, 4, 0x01000303> yxww, graa;
      swizzle<T, 4, 4, 0x01010000> yyxx, ggrr;
      swizzle<T, 4, 4, 0x01010001> yyxy, ggrg;
      swizzle<T, 4, 4, 0x01010002> yyxz, ggrb;
      swizzle<T, 4, 4, 0x01010003> yyxw, ggra;
      swizzle<T, 4, 4, 0x01010100> yyyx, gggr;
      swizzle<T, 4, 4, 0x01010101> yyyy, gggg;
      swizzle<T, 4, 4, 0x01010102> yyyz, gggb;
      swizzle<T, 4, 4, 0x01010103> yyyw, ggga;
      swizzle<T, 4, 4, 0x01010200> yyzx, ggbr;
      swizzle<T, 4, 4, 0x01010201> yyzy, ggbg;
      swizzle<T, 4, 4, 0x01010202> yyzz, ggbb;
      swizzle<T, 4, 4, 0x01010203> yyzw, ggba;
      swizzle<T, 4, 4, 0x01010300> yywx, ggar;
      swizzle<T, 4, 4, 0x01010301> yywy, ggag;
      swizzle<T, 4, 4, 0x01010302> yywz, ggab;
      swizzle<T, 4, 4, 0x01010303> yyww, ggaa;
      swizzle<T, 4, 4, 0x01020000> yzxx, gbrr;
      swizzle<T, 4, 4, 0x01020001> yzxy, gbrg;
      swizzle<T, 4, 4, 0x01020002> yzxz, gbrb;
      swizzle<T, 4, 4, 0x01020003> yzxw, gbra;
      swizzle<T, 4, 4, 0x01020100> yzyx, gbgr;
      swizzle<T, 4, 4, 0x01020101> yzyy, gbgg;
      swizzle<T, 4, 4, 0x01020102> yzyz, gbgb;
      swizzle<T, 4, 4, 0x01020103> yzyw, gbga;
      swizzle<T, 4, 4, 0x01020200> yzzx, gbbr;
      swizzle<T, 4, 4, 0x01020201> yzzy, gbbg;
      swizzle<T, 4, 4, 0x01020202> yzzz, gbbb;
      swizzle<T, 4, 4, 0x01020203> yzzw, gbba;
      swizzle<T, 4, 4, 0x01020300> yzwx, gbar;
      swizzle<T, 4, 4, 0x01020301> yzwy, gbag;
      swizzle<T, 4, 4, 0x01020302> yzwz, gbab;
      swizzle<T, 4, 4, 0x01020303> yzww, gbaa;
      swizzle<T, 4, 4, 0x01030000> ywxx, garr;
      swizzle<T, 4, 4, 0x01030001> ywxy, garg;
      swizzle<T, 4, 4, 0x01030002> ywxz, garb;
      swizzle<T, 4, 4, 0x01030003> ywxw, gara;
      swizzle<T, 4, 4, 0x01030100> ywyx, gagr;
      swizzle<T, 4, 4, 0x01030101> ywyy, gagg;
      swizzle<T, 4, 4, 0x01030102> ywyz, gagb;
      swizzle<T, 4, 4, 0x01030103> ywyw, gaga;
      swizzle<T, 4, 4, 0x01030200> ywzx, gabr;
      swizzle<T, 4, 4, 0x01030201> ywzy, gabg;
      swizzle<T, 4, 4, 0x01030202> ywzz, gabb;
      swizzle<T, 4, 4, 0x01030203> ywzw, gaba;
      swizzle<T, 4, 4, 0x01030300> ywwx, gaar;
      swizzle<T, 4, 4, 0x01030301> ywwy, gaag;
      swizzle<T, 4, 4, 0x01030302> ywwz, gaab;
      swizzle<T, 4, 4, 0x01030303> ywww, gaaa;
      swizzle<T, 4, 4, 0x02000000> zxxx, brrr;
      swizzle<T, 4, 4, 0x02000001> zxxy, brrg;
      swizzle<T, 4, 4, 0x02000002> zxxz, brrb;
      swizzle<T, 4, 4, 0x02000003> zxxw, brra;
      swizzle<T, 4, 4, 0x02000100> zxyx, brgr;
      swizzle<T, 4, 4, 0x02000101> zxyy, brgg;
      swizzle<T, 4, 4, 0x02000102> zxyz, brgb;
      swizzle<T, 4, 4, 0x02000103> zxyw, brga;
      swizzle<T, 4, 4, 0x02000200> zxzx, brbr;
      swizzle<T, 4, 4, 0x02000201> zxzy, brbg;
      swizzle<T, 4, 4, 0x02000202> zxzz, brbb;
      swizzle<T, 4, 4, 0x02000203> zxzw, brba;
      swizzle<T, 4, 4, 0x02000300> zxwx, brar;
      swizzle<T, 4, 4, 0x02000301> zxwy, brag;
      swizzle<T, 4, 4, 0x02000302> zxwz, brab;
      swizzle<T, 4, 4, 0x02000303> zxww, braa;
      swizzle<T, 4, 4, 0x02010000> zyxx, bgrr;
      swizzle<T, 4, 4, 0x02010001> zyxy, bgrg;
      swizzle<T, 4, 4, 0x02010002> zyxz, bgrb;
      swizzle<T, 4, 4, 0x02010003> zyxw, bgra;
      swizzle<T, 4, 4, 0x02010100> zyyx, bggr;
      swizzle<T, 4, 4, 0x02010101> zyyy, bggg;
      swizzle<T, 4, 4, 0x02010102> zyyz, bggb;
      swizzle<T, 4, 4, 0x02010103> zyyw, bgga;
      swizzle<T, 4, 4, 0x02010200> zyzx, bgbr;
      swizzle<T, 4, 4, 0x02010201> zyzy, bgbg;
      swizzle<T, 4, 4, 0x02010202> zyzz, bgbb;
      swizzle<T, 4, 4, 0x02010203> zyzw, bgba;
      swizzle<T, 4, 4, 0x02010300> zywx, bgar;
      swizzle<T, 4, 4, 0x02010301> zywy, bgag;
      swizzle<T, 4, 4, 0x02010302> zywz, bgab;
      swizzle<T, 4, 4, 0x02010303> zyww, bgaa;
      swizzle<T, 4, 4, 0x02020000> zzxx, bbrr;
      swizzle<T, 4, 4, 0x02020001> zzxy, bbrg;
      swizzle<T, 4, 4, 0x02020002> zzxz, bbrb;
      swizzle<T, 4, 4, 0x02020003> zzxw, bbra;
      swizzle<T, 4, 4, 0x02020100> zzyx, bbgr;
      swizzle<T, 4, 4, 0x02020101> zzyy, bbgg;
      swizzle<T, 4, 4, 0x02020102> zzyz, bbgb;
      swizzle<T, 4, 4, 0x02020103> zzyw, bbga;
      swizzle<T, 4, 4, 0x02020200> zzzx, bbbr;
      swizzle<T, 4, 4, 0x02020201> zzzy, bbbg;
      swizzle<T, 4, 4, 0x02020202> zzzz, bbbb;
      swizzle<T, 4, 4, 0x02020203> zzzw, bbba;
      swizzle<T, 4, 4, 0x02020300> zzwx, bbar;
      swizzle<T, 4, 4, 0x02020301> zzwy, bbag;
      swizzle<T, 4, 4, 0x02020302> zzwz, bbab;
      swizzle<T, 4, 4, 0x02020303> zzww, bbaa;
      swizzle<T, 4, 4, 0x02030000> zwxx, barr;
      swizzle<T, 4, 4, 0x02030001> zwxy, barg;
      swizzle<T, 4, 4, 0x02030002> zwxz, barb;
      swizzle<T, 4, 4, 0x02030003> zwxw, bara;
      swizzle<T, 4, 4, 0x02030100> zwyx, bagr;
      swizzle<T, 4, 4, 0x02030101> zwyy, bagg;
      swizzle<T, 4, 4, 0x02030102> zwyz, bagb;
      swizzle<T, 4, 4, 0x02030103> zwyw, baga;
      swizzle<T, 4, 4, 0x02030200> zwzx, babr;
      swizzle<T, 4, 4, 0x02030201> zwzy, babg;
      swizzle<T, 4, 4, 0x02030202> zwzz, babb;
      swizzle<T, 4, 4, 0x02030203> zwzw, baba;
      swizzle<T, 4, 4, 0x02030300> zwwx, baar;
      swizzle<T, 4, 4, 0x02030301> zwwy, baag;
      swizzle<T, 4, 4, 0x02030302> zwwz, baab;
      swizzle<T, 4, 4, 0x02030303> zwww, baaa;
      swizzle<T, 4, 4, 0x03000000> wxxx, arrr;
      swizzle<T, 4, 4, 0x03000001> wxxy, arrg;
      swizzle<T, 4, 4, 0x03000002> wxxz, arrb;
      swizzle<T, 4, 4, 0x03000003> wxxw, arra;
      swizzle<T, 4, 4, 0x03000100> wxyx, argr;
      swizzle<T, 4, 4, 0x03000101> wxyy, argg;
      swizzle<T, 4, 4, 0x03000102> wxyz, argb;
      swizzle<T, 4, 4, 0x03000103> wxyw, arga;
      swizzle<T, 4, 4, 0x03000200> wxzx, arbr;
      swizzle<T, 4, 4, 0x03000201> wxzy, arbg;
      swizzle<T, 4, 4, 0x03000202> wxzz, arbb;
      swizzle<T, 4, 4, 0x03000203> wxzw, arba;
      swizzle<T, 4, 4, 0x03000300> wxwx, arar;
      swizzle<T, 4, 4, 0x03000301> wxwy, arag;
      swizzle<T, 4, 4, 0x03000302> wxwz, arab;
      swizzle<T, 4, 4, 0x03000303> wxww, araa;
      swizzle<T, 4, 4, 0x03010000> wyxx, agrr;
      swizzle<T, 4, 4, 0x03010001> wyxy, agrg;
      swizzle<T, 4, 4, 0x03010002> wyxz, agrb;
      swizzle<T, 4, 4, 0x03010003> wyxw, agra;
      swizzle<T, 4, 4, 0x03010100> wyyx, aggr;
      swizzle<T, 4, 4, 0x03010101> wyyy, aggg;
      swizzle<T, 4, 4, 0x03010102> wyyz, aggb;
      swizzle<T, 4, 4, 0x03010103> wyyw, agga;
      swizzle<T, 4, 4, 0x03010200> wyzx, agbr;
      swizzle<T, 4, 4, 0x03010201> wyzy, agbg;
      swizzle<T, 4, 4, 0x03010202> wyzz, agbb;
      swizzle<T, 4, 4, 0x03010203> wyzw, agba;
      swizzle<T, 4, 4, 0x03010300> wywx, agar;
      swizzle<T, 4, 4, 0x03010301> wywy, agag;
      swizzle<T, 4, 4, 0x03010302> wywz, agab;
      swizzle<T, 4, 4, 0x03010303> wyww, agaa;
      swizzle<T, 4, 4, 0x03020000> wzxx, abrr;
      swizzle<T, 4, 4, 0x03020001> wzxy, abrg;
      swizzle<T, 4, 4, 0x03020002> wzxz, abrb;
      swizzle<T, 4, 4, 0x03020003> wzxw, abra;
      swizzle<T, 4, 4, 0x03020100> wzyx, abgr;
      swizzle<T, 4, 4, 0x03020101> wzyy, abgg;
      swizzle<T, 4, 4, 0x03020102> wzyz, abgb;
      swizzle<T, 4, 4, 0x03020103> wzyw, abga;
      swizzle<T, 4, 4, 0x03020200> wzzx, abbr;
      swizzle<T, 4, 4, 0x03020201> wzzy, abbg;
      swizzle<T, 4, 4, 0x03020202> wzzz, abbb;
      swizzle<T, 4, 4, 0x03020203> wzzw, abba;
      swizzle<T, 4, 4, 0x03020300> wzwx, abar;
      swizzle<T, 4, 4, 0x03020301> wzwy, abag;
      swizzle<T, 4, 4, 0x03020302> wzwz, abab;
      swizzle<T, 4, 4, 0x03020303> wzww, abaa;
      swizzle<T, 4, 4, 0x03030000> wwxx, aarr;
      swizzle<T, 4, 4, 0x03030001> wwxy, aarg;
      swizzle<T, 4, 4, 0x03030002> wwxz, aarb;
      swizzle<T, 4, 4, 0x03030003> wwxw, aara;
      swizzle<T, 4, 4, 0x03030100> wwyx, aagr;
      swizzle<T, 4, 4, 0x03030101> wwyy, aagg;
      swizzle<T, 4, 4, 0x03030102> wwyz, aagb;
      swizzle<T, 4, 4, 0x03030103> wwyw, aaga;
      swizzle<T, 4, 4, 0x03030200> wwzx, aabr;
      swizzle<T, 4, 4, 0x03030201> wwzy, aabg;
      swizzle<T, 4, 4, 0x03030202> wwzz, aabb;
      swizzle<T, 4, 4, 0x03030203> wwzw, aaba;
      swizzle<T, 4, 4, 0x03030300> wwwx, aaar;
      swizzle<T, 4, 4, 0x03030301> wwwy, aaag;
      swizzle<T, 4, 4, 0x03030302> wwwz, aaab;
      swizzle<T, 4, 4, 0x03030303> wwww, aaaa;
    };
  };

  template <typename T, size_t N>
  inline vector<T, N> sin(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::sin(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> cos(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::cos(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> tan(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::tan(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> asin(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::asin(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> acos(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::acos(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> atan(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::atan(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> pow(const vector<T, N> &x, const vector<T, N> &y) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::pow(x[i], y[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> exp(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::exp(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> log(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::log(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> exp2(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::exp2(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> log2(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::log2(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> sqrt(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::sqrt(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> rsqrt(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return T(1) / std::sqrt(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> abs(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::abs(v[i]); });
  }
  template <typename T> inline T sign(const T &t) {
    return (T(0) < t) - (t < T(0));
  }
  template <typename T, size_t N>
  inline vector<T, N> sign(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return ::vml::detail::sign(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> floor(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::floor(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> trunc(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::trunc(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> ceil(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return std::ceil(v[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> fract(const vector<T, N> &v) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return (v[i] - std::floor(v[i])); });
  }
  template <typename T, size_t N>
  inline vector<T, N> mod(const vector<T, N> &x, const T &y) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return x[i] - y * std::floor(x[i] / y); });
  }
  template <typename T, size_t N>
  inline vector<T, N> mod(const vector<T, N> &x, const vector<T, N> &y) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return x[i] - y[i] * std::floor(x[i] / y[i]); });
  }
  template <typename T, size_t N>
  inline vector<T, N> min(const vector<T, N> &x, const T &y) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return x[i] < y ? x[i] : y; });
  }
  template <typename T, size_t N>
  inline vector<T, N> min(const vector<T, N> &x, const vector<T, N> &y) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return x[i] < y[i] ? x[i] : y[i]; });
  }
  template <typename T, size_t N>
  inline vector<T, N> max(const vector<T, N> &x, const T &y) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return x[i] > y ? x[i] : y; });
  }
  template <typename T, size_t N>
  inline vector<T, N> max(const vector<T, N> &x, const vector<T, N> &y) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return x[i] > y[i] ? x[i] : y[i]; });
  }
  template <typename T, size_t N>
  inline vector<T, N> clamp(const vector<T, N> &x, const T &min_val,
                            const T &max_val) {
    return min(max(x, min_val), max_val);
  }
  template <typename T, size_t N>
  inline vector<T, N> clamp(const vector<T, N> &x, const vector<T, N> &min_val,
                            const vector<T, N> &max_val) {
    return min(max(x, min_val), max_val);
  }
  template <typename T, size_t N>
  inline vector<T, N> mix(const vector<T, N> &x, const vector<T, N> &y,
                          const T &a) {
    return x * (T(1) - a) + y * a;
  }
  template <typename T, size_t N>
  inline vector<T, N> mix(const vector<T, N> &x, const vector<T, N> &y,
                          const vector<T, N> &a) {
    return x * (T(1) - a) + y * a;
  }
  template <typename T, size_t N>
  inline vector<T, N> step(const T &edge, const vector<T, N> &x) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return x[i] < edge ? T(0) : T(1); });
  }
  template <typename T, size_t N>
  inline vector<T, N> step(const vector<T, N> &edge, const vector<T, N> &x) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return x[i] < edge[i] ? T(0) : T(1); });
  }
  template <typename T, size_t N>
  inline vector<T, N> smoothstep(const T &edge0, const T &edge1,
                                 const vector<T, N> &x) {
    auto t = clamp((x - edge0) / (edge1 - edge0), T(0), T(1));
    return t * t * (T(3) - T(2) * t);
  }
  template <typename T, size_t N>
  inline vector<T, N> smoothstep(const vector<T, N> &edge0,
                                 const vector<T, N> &edge1,
                                 const vector<T, N> &x) {
    auto t = clamp((x - edge0) / (edge1 - edge0), T(0), T(1));
    return t * t * (T(3) - T(2) * t);
  }

  template <typename T, size_t N> inline T length(const vector<T, N> &v) {
    return std::sqrt(dot(v, v));
  }
  template <typename T, size_t N>
  inline T distance(const vector<T, N> &p0, const vector<T, N> &p1) {
    return length(p0 - p1);
  }
  template <typename T, size_t N>
  inline vector<T, N> normalize(const vector<T, N> &v) {
    return v / length(v);
  }
  template <typename T, size_t N>
  inline T dot(const vector<T, N> &a, const vector<T, N> &b) {
    T sum = 0;
    ::vml::detail::static_for<0, N>([&](size_t i) { sum += a[i] * b[i]; });
    return sum;
  }
  template <typename T>
  inline vector<T, 3> cross(const vector<T, 3> &a, const vector<T, 3> &b) {
    return vector<T, 3>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x);
  }
  template <typename T, size_t Size>
  inline vector<T, Size> faceforward(const vector<T, Size> &N,
                                     const vector<T, Size> &I,
                                     const vector<T, Size> &Nref) {
    return dot(Nref, I) < T(0) ? N : (-N);
  }
  template <typename T, size_t Size>
  inline vector<T, Size> reflect(const vector<T, Size> &I,
                                 const vector<T, Size> &N) {
    return (I - T(2) * dot(I, N) * N);
  }
  template <typename T, size_t Size>
  inline vector<T, Size> refract(const vector<T, Size> &I,
                                 const vector<T, Size> &N, const T &eta) {
    T k = T(1) - eta * eta * (T(1) - dot(N, I) * dot(N, I));
    if (k < T(0)) {
      return vector<T, Size>();
    } else {
      return eta * I - (eta * dot(N, I) + std::sqrt(k)) * N;
    }
  }

  template <typename T, size_t N, typename CharT = char>
  std::basic_string<CharT> fmt(const vector<T, N> &v) {
    std::basic_string<CharT> out = "(" + std::to_string(v[0]);
    for (size_t i = 1; i < N; ++i) {
      out += ',' + std::to_string(v[i]);
    }
    return out + ")";
  }
  template <typename T, size_t M, size_t N, typename CharT = char>
  std::basic_string<CharT> fmt(const matrix<T, M, N> &m) {
    std::basic_string<CharT> out = "[[" + std::to_string(m[0][0]);
    for (size_t c = 1; c < N; ++c) {
      out += ',' + std::to_string(m[0][c]);
    }
    out += ']';
    for (size_t r = 1; r < M; ++r) {
      out += ",[" + std::to_string(m[r][0]);
      for (size_t c = 1; c < N; ++c) {
        out += ',' + std::to_string(m[r][c]);
      }
      out += ']';
    }
    return out + "]";
  }

} // namespace detail

template <typename T, size_t N>
struct vector : public ::vml::detail::vector_base<T, N> {
  typedef T scalar_type;
  typedef vector<T, N> vector_type;
  typedef typename ::vml::detail::vector_base<T, N> base_type;
  typedef typename std::conditional<N == 1, scalar_type, vector_type>::type
      decay_type;

  using base_type::data;

  vector() { std::fill(data, data + N, scalar_type(0)); };
  vector(const scalar_type *swizzle_data) {
    std::copy(swizzle_data, swizzle_data + N, data);
  }
  vector(typename std::conditional<N == 1, scalar_type,
                                   ::vml::detail::nothing>::type s) {
    data[0] = s;
  }
  vector(typename std::conditional<N != 1, scalar_type,
                                   ::vml::detail::nothing>::type s) {
    std::fill(data, data + N, s);
  }
  template <typename A0, typename... Args,
            class = typename std::enable_if<(sizeof...(Args) >= 1 &&
                                             sizeof...(Args) < N)>::type>
  explicit vector(A0 &&a0, Args &&... args) {
    __construct<0>(::vml::detail::decay(std::forward<A0>(a0)),
                   ::vml::detail::decay(std::forward<Args>(args))...);
  }

  scalar_type &operator[](size_t i) { return data[i]; }
  const scalar_type &operator[](size_t i) const { return data[i]; }
  decay_type decay() const { return static_cast<const decay_type &>(*this); }
  operator typename std::conditional<N == 1, scalar_type,
                                     ::vml::detail::nothing>::type() const {
    return data[0];
  }

  vector_type operator-() {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return -data[i]; });
  }
  vector_type &operator+=(scalar_type s) {
    ::vml::detail::static_for<0, N>([&](size_t i) { data[i] += s; });
    return *this;
  }
  vector_type &operator-=(scalar_type s) {
    ::vml::detail::static_for<0, N>([&](size_t i) { data[i] -= s; });
    return *this;
  }
  vector_type &operator*=(scalar_type s) {
    ::vml::detail::static_for<0, N>([&](size_t i) { data[i] *= s; });
    return *this;
  }
  vector_type &operator/=(scalar_type s) {
    ::vml::detail::static_for<0, N>([&](size_t i) { data[i] /= s; });
    return *this;
  }
  vector_type &operator+=(vector_type s) {
    ::vml::detail::static_for<0, N>([&](size_t i) { data[i] += s[i]; });
    return *this;
  }
  vector_type &operator-=(vector_type s) {
    ::vml::detail::static_for<0, N>([&](size_t i) { data[i] -= s[i]; });
    return *this;
  }
  vector_type &operator*=(vector_type s) {
    ::vml::detail::static_for<0, N>([&](size_t i) { data[i] *= s[i]; });
    return *this;
  }
  vector_type &operator/=(vector_type s) {
    ::vml::detail::static_for<0, N>([&](size_t i) { data[i] /= s[i]; });
    return *this;
  }

  friend vector_type operator+(const vector_type &a, const scalar_type &b) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return a[i] + b; });
  }
  friend vector_type operator-(const vector_type &a, const scalar_type &b) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return a[i] - b; });
  }
  friend vector_type operator*(const vector_type &a, const scalar_type &b) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return a[i] * b; });
  }
  friend vector_type operator/(const vector_type &a, const scalar_type &b) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return a[i] / b; });
  }
  friend vector_type operator+(const scalar_type &a, const vector_type &b) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return a + b[i]; });
  }
  friend vector_type operator-(const scalar_type &a, const vector_type &b) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return a - b[i]; });
  }
  friend vector_type operator*(const scalar_type &a, const vector_type &b) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return a * b[i]; });
  }
  friend vector_type operator/(const scalar_type &a, const vector_type &b) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return a / b[i]; });
  }
  friend vector_type operator+(const vector_type &a, const vector_type &b) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return a[i] + b[i]; });
  }
  friend vector_type operator-(const vector_type &a, const vector_type &b) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return a[i] - b[i]; });
  }
  friend vector_type operator*(const vector_type &a, const vector_type &b) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return a[i] * b[i]; });
  }
  friend vector_type operator/(const vector_type &a, const vector_type &b) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return a[i] / b[i]; });
  }

private:
  template <size_t I>
  typename std::enable_if<(I < N), void>::type __construct() {
    std::fill(data + I, data + N, scalar_type(0));
  }
  template <size_t I>
  typename std::enable_if<(I >= N), void>::type __construct() {}
  template <size_t I> void __construct(scalar_type arg) {
    data[I] = arg;
    __construct<I + 1>();
  }
  template <size_t I, typename TOther, size_t NOther>
  void __construct(const vector<TOther, NOther> &arg) {
    std::copy<scalar_type>(arg.data, arg.data + NOther, data + I);
    __construct<I + NOther>();
  }
  template <size_t I, typename... Args>
  void __construct(scalar_type arg, Args &&... args) {
    data[I] = arg;
    __construct<I + 1>(args...);
  }
  template <size_t I, typename TOther, size_t NOther, typename... Args>
  void __construct(const vector<TOther, NOther> &arg, Args &&... args) {
    std::copy<scalar_type>(arg.data, arg.data + NOther, data + I);
    __construct<I + NOther>(args...);
  }
};

template <typename T, size_t M, size_t N> struct matrix {
  typedef T scalar_type;
  typedef vector<T, N> vector_type;
  typedef vector<T, M> row_type;
  typedef vector<T, N> column_type;
  matrix() = default;

  template <typename S, typename = typename std::enable_if<
                            std::is_convertible<S, T>::value && (N == M)>::type>
  explicit matrix(S s) {
    ::vml::detail::static_for<0, M>([&](size_t i) { data[i][i] = s; });
  }
  matrix(const matrix &other) {
    ::vml::detail::static_for<0, M>([&](size_t i) { data[i] = other[i]; });
  }
  template <size_t OtherM, size_t OtherN>
  matrix(const matrix<T, OtherM, OtherN> &other) {
    static constexpr auto MinM = M > OtherM ? OtherM : M;
    static constexpr auto MinN = N > OtherN ? OtherN : N;
    static constexpr auto MinInner = MinN > MinM ? MinM : MinN;
    static constexpr auto MinOuter = N > M ? M : N;
    ::vml::detail::static_for<0, MinM>([&](size_t row) {
      ::vml::detail::static_for<0, MinN>(
          [&](size_t col) { data[row][col] = other[row][col]; });
    });
    ::vml::detail::static_for<MinInner, MinOuter>(
        [&](size_t i) { data[i][i] = T(1); });
  }
  template <typename A0, typename... Args,
            class = typename std::enable_if<(sizeof...(Args) >= 1 &&
                                             sizeof...(Args) < N * M)>::type>
  explicit matrix(A0 a0, Args &&... args) {
    __construct<0>(::vml::detail::decay(std::forward<A0>(a0)),
                   ::vml::detail::decay(std::forward<Args>(args))...);
  }

  matrix decay() const { return *this; }

  row_type &operator[](size_t i) { return data[i]; }
  const row_type &operator[](size_t i) const { return data[i]; }

  column_type column(size_t i) const {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t j) { return data[j][i]; });
  }
  row_type row(size_t i) const { return data[i]; }

  matrix operator-() {
    return ::vml::detail::static_constructor<T, N, M>(
        [&](size_t i) { return -data[i]; });
  }
  matrix &operator+=(scalar_type s) {
    ::vml::detail::static_for<0, M>([&](size_t i) { data[i] += s; });
    return *this;
  }
  matrix &operator-=(scalar_type s) {
    ::vml::detail::static_for<0, M>([&](size_t i) { data[i] -= s; });
    return *this;
  }
  matrix &operator*=(scalar_type s) {
    ::vml::detail::static_for<0, M>([&](size_t i) { data[i] *= s; });
    return *this;
  }
  matrix &operator/=(scalar_type s) {
    ::vml::detail::static_for<0, M>([&](size_t i) { data[i] /= s; });
    return *this;
  }
  matrix &operator+=(matrix s) {
    ::vml::detail::static_for<0, M>([&](size_t i) { data[i] += s[i]; });
    return *this;
  }
  matrix &operator-=(matrix s) {
    ::vml::detail::static_for<0, M>([&](size_t i) { data[i] -= s[i]; });
    return *this;
  }
  matrix &operator/=(matrix s) {
    ::vml::detail::static_for<0, M>([&](size_t i) { data[i] /= s[i]; });
    return *this;
  }

  friend matrix operator+(const matrix &a, const scalar_type &b) {
    return ::vml::detail::static_constructor<T, M, N>(
        [&](size_t i) { return a[i] + b; });
  }
  friend matrix operator-(const matrix &a, const scalar_type &b) {
    return ::vml::detail::static_constructor<T, M, N>(
        [&](size_t i) { return a[i] - b; });
  }
  friend matrix operator*(const matrix &a, const scalar_type &b) {
    return ::vml::detail::static_constructor<T, M, N>(
        [&](size_t i) { return a[i] * b; });
  }
  friend matrix operator/(const matrix &a, const scalar_type &b) {
    return ::vml::detail::static_constructor<T, M, N>(
        [&](size_t i) { return a[i] / b; });
  }
  friend matrix operator+(const scalar_type &a, const matrix &b) {
    return ::vml::detail::static_constructor<T, M, N>(
        [&](size_t i) { return a + b[i]; });
  }
  friend matrix operator-(const scalar_type &a, const matrix &b) {
    return ::vml::detail::static_constructor<T, M, N>(
        [&](size_t i) { return a - b[i]; });
  }
  friend matrix operator*(const scalar_type &a, const matrix &b) {
    return ::vml::detail::static_constructor<T, M, N>(
        [&](size_t i) { return a * b[i]; });
  }
  friend matrix operator/(const scalar_type &a, const matrix &b) {
    return ::vml::detail::static_constructor<T, M, N>(
        [&](size_t i) { return a / b[i]; });
  }
  friend matrix operator+(const matrix &a, const matrix &b) {
    return ::vml::detail::static_constructor<T, M, N>(
        [&](size_t i) { return a[i] + b[i]; });
  }
  friend matrix operator-(const matrix &a, const matrix &b) {
    return ::vml::detail::static_constructor<T, M, N>(
        [&](size_t i) { return a[i] - b[i]; });
  }
  friend matrix operator/(const matrix &a, const matrix &b) {
    return ::vml::detail::static_constructor<T, M, N>(
        [&](size_t i) { return a[i] / b[i]; });
  }

  friend column_type operator*(const matrix &m, const row_type &v) {
    return mul(m, v);
  }
  friend row_type operator*(const column_type &v, const matrix &m) {
    return mul(v, m);
  }
  matrix &operator*=(const matrix &m) { return *this = mul(*this, m); }
  template <size_t OtherM>
  friend matrix<T, OtherM, N> operator*(const matrix &m1,
                                        const matrix<T, OtherM, N> &m2) {
    return mul(m1, m2);
  }

  static column_type mul(const matrix &m, const row_type &v) {
    return ::vml::detail::static_constructor<T, M>(
        [&](size_t i) { return ::vml::detail::dot(v, m.row(i)); });
  }
  static row_type mul(const column_type &v, const matrix &m) {
    return ::vml::detail::static_constructor<T, N>(
        [&](size_t i) { return ::vml::detail::dot(v, m.column(i)); });
  }
  template <size_t OtherM>
  static matrix<T, OtherM, N> mul(const matrix &m1,
                                  const matrix<T, OtherM, N> &m2) {
    matrix<T, OtherM, N> out;
    ::vml::detail::static_for<0, N>([&](size_t i) { out[i] = m1.row(i) * m2; });
    return out;
  }

  row_type data[M];

private:
  template <size_t I>
  typename std::enable_if<(I < M * N), void>::type __construct() {}
  template <size_t I>
  typename std::enable_if<(I >= M * N), void>::type __construct() {}
  template <size_t I> void __construct(scalar_type arg) {
    data[I / N][I % N] = arg;
    __construct<I + 1>();
  }
  template <size_t I, typename TOther, size_t NOther>
  void __construct(const vector<TOther, NOther> &arg) {
    ::vml::detail::static_for<0, NOther>(
        [&](size_t i) { data[(I + i) / N][(I + i) % N] = arg[i]; });
    __construct<I + NOther>();
  }
  template <size_t I, typename... Args>
  void __construct(scalar_type arg, Args &&... args) {
    data[I / N][I % N] = arg;
    __construct<I + 1>(args...);
  }
  template <size_t I, typename TOther, size_t NOther, typename... Args>
  void __construct(const vector<TOther, NOther> &arg, Args &&... args) {
    ::vml::detail::static_for<0, NOther>(
        [&](size_t i) { data[(I + i) / N][(I + i) % N] = arg[i]; });
    __construct<I + NOther>(args...);
  }
};

VML_FUNC(sin);
VML_FUNC(cos);
VML_FUNC(tan);
VML_FUNC(asin);
VML_FUNC(acos);
VML_FUNC(atan);
VML_FUNC(pow);
VML_FUNC(exp);
VML_FUNC(log);
VML_FUNC(exp2);
VML_FUNC(log2);
VML_FUNC(sqrt);
VML_FUNC(rsqrt);
VML_FUNC(abs);
VML_FUNC(sign);
VML_FUNC(floor);
VML_FUNC(trunc);
VML_FUNC(ceil);
VML_FUNC(fract);
VML_FUNC(mod);
VML_FUNC(min);
VML_FUNC(max);
VML_FUNC(clamp);
VML_FUNC(mix);
VML_FUNC(step);
VML_FUNC(smoothstep);
VML_FUNC(length);
VML_FUNC(distance);
VML_FUNC(normalize);
VML_FUNC(dot);
VML_FUNC(cross);
VML_FUNC(faceforward);
VML_FUNC(reflect);
VML_FUNC(refract);

VML_FUNC(fmt);

template <typename T>
inline constexpr matrix<T, 4, 4> look_at(const vector<T, 3> &eye,
                                         const vector<T, 3> &center,
                                         const vector<T, 3> &up) {
  const vector<T, 3> f(normalize(center - eye));
  const vector<T, 3> s(normalize(cross(up, f)));
  const vector<T, 3> u(cross(f, s));

  matrix<T, 4, 4> result(1);
  result[0][0] = s.x;
  result[0][1] = s.y;
  result[0][2] = s.z;
  result[1][0] = u.x;
  result[1][1] = u.y;
  result[1][2] = u.z;
  result[2][0] = f.x;
  result[2][1] = f.y;
  result[2][2] = f.z;
  result[0][3] = -dot(s, eye);
  result[1][3] = -dot(u, eye);
  result[1][3] = -dot(f, eye);
  return result;
}

template <typename T> using tmat4 = ::vml::matrix<T, 4, 4>;
template <typename T> using tmat3 = ::vml::matrix<T, 3, 3>;
template <typename T> using tmat2 = ::vml::matrix<T, 2, 2>;
template <typename T> using tvec4 = ::vml::vector<T, 4>;
template <typename T> using tvec3 = ::vml::vector<T, 3>;
template <typename T> using tvec2 = ::vml::vector<T, 2>;

typedef tmat4<float> fmat4;
typedef tmat3<float> fmat3;
typedef tmat2<float> fmat2;
typedef tmat4<bool> bmat4;
typedef tmat3<bool> bmat3;
typedef tmat2<bool> bmat2;
typedef tmat4<double> dmat4;
typedef tmat3<double> dmat3;
typedef tmat2<double> dmat2;
typedef tmat4<int> imat4;
typedef tmat3<int> imat3;
typedef tmat2<int> imat2;
typedef tmat4<unsigned> umat4;
typedef tmat3<unsigned> umat3;
typedef tmat2<unsigned> umat2;

typedef tvec4<float> fvec4;
typedef tvec3<float> fvec3;
typedef tvec2<float> fvec2;
typedef tvec4<bool> bvec4;
typedef tvec3<bool> bvec3;
typedef tvec2<bool> bvec2;
typedef tvec4<double> dvec4;
typedef tvec3<double> dvec3;
typedef tvec2<double> dvec2;
typedef tvec4<int> ivec4;
typedef tvec3<int> ivec3;
typedef tvec2<int> ivec2;
typedef tvec4<unsigned> uvec4;
typedef tvec3<unsigned> uvec3;
typedef tvec2<unsigned> uvec2;

typedef fmat4 mat4;
typedef fmat3 mat3;
typedef fmat2 mat2;
typedef fvec4 vec4;
typedef fvec3 vec3;
typedef fvec2 vec2;

} // namespace vml

#endif // VML_HPP_
