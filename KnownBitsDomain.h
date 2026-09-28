//===- KnownBitsDomain.h - The abstract domain ----------------------------===//
//
// A four-point lattice recording whether an integer value is known to be zero.
//
//        Top          nothing is known
//       /   \
//    Zero  NonZero
//       \   /
//       Bottom       unreachable, or not yet analyzed
//
// This is the file to replace first when building a different analysis.  MLIR's
// dataflow framework asks only three things of a lattice value:
//
//   * a default constructor, which must produce the bottom element, because the
//     solver starts every value optimistically and lowers it as facts arrive;
//   * a static join(), which must be commutative, associative, idempotent, and
//     monotone -- assertions in Lattice<> check monotonicity in debug builds;
//   * operator== and print().
//
//===----------------------------------------------------------------------===//

#ifndef ZERO_DOMAIN_H
#define ZERO_DOMAIN_H

#include "llvm/Support/raw_ostream.h"

namespace known_bits {

enum class Kind { Bottom, Zero, NonZero, Top };

inline const char *name(Kind kind) {
  switch (kind) {
  case Kind::Bottom:
    return "bottom";
  case Kind::Zero:
    return "zero";
  case Kind::NonZero:
    return "nonzero";
  case Kind::Top:
    return "top";
  }
  return "top";
}

struct KnownBitsState {
  Kind kind = Kind::Bottom;

  KnownBitsState() = default;
  /* implicit */ KnownBitsState(Kind kind) : kind(kind) {}

  static KnownBitsState bottom() { return Kind::Bottom; }
  static KnownBitsState top() { return Kind::Top; }

  bool isBottom() const { return kind == Kind::Bottom; }

  /// Least upper bound.  Two disagreeing facts lose all information.
  static KnownBitsState join(const KnownBitsState &lhs, const KnownBitsState &rhs) {
    if (lhs.kind == Kind::Bottom)
      return rhs;
    if (rhs.kind == Kind::Bottom)
      return lhs;
    if (lhs.kind == rhs.kind)
      return lhs;
    return top();
  }

  bool operator==(const KnownBitsState &other) const { return kind == other.kind; }
  bool operator!=(const KnownBitsState &other) const { return kind != other.kind; }

  void print(llvm::raw_ostream &os) const { os << name(kind); }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const KnownBitsState &state) {
  state.print(os);
  return os;
}

} // namespace known_bits

#endif
