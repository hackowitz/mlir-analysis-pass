//===- KnownBitsDomain.h - The abstract domain ----------------------------===//
//
// A four-point lattice recording whether a bit value is known to be zero.
//
//        Top          nothing is known
//       /   \
//    Zero  One
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

#ifndef KNOWN_BITS_DOMAIN_H
#define KNOWN_BITS_DOMAIN_H

#include "llvm/Support/raw_ostream.h"

namespace known_bits {

struct KnownBitsState {
  /// @brief bit mask of abstract states for each bit.
  /// - "bottom" occurs when neigher can_be_one or can_be_zero.
  /// - "top" occurs when both can_be_one and can_be_zero.
  /// The default is therefore "bottom" for all bits
  unsigned int can_be_one = 0;
  unsigned int can_be_zero = 0;

  KnownBitsState() = default;
  KnownBitsState(unsigned int one, unsigned int zero) : can_be_one(one), can_be_zero(zero) {}
  KnownBitsState bottom() { return KnownBitsState(0, 0); }
  KnownBitsState top() { return KnownBitsState(~0, ~0); }

  bool isBottom() const { return ~(can_be_one | can_be_zero); }
  bool isTop() const { return (can_be_one & can_be_zero); }

  /// @brief A 'join' is a logical or.
  static KnownBitsState join(const KnownBitsState &lhs, const KnownBitsState &rhs) {
    return KnownBitsState(lhs.can_be_one | rhs.can_be_one, lhs.can_be_zero | rhs.can_be_zero);
  }

  bool operator==(const KnownBitsState &other) const {
    return (can_be_one == other.can_be_one) && (can_be_zero == other.can_be_zero);
  }
  bool operator!=(const KnownBitsState &other) const {
    return (can_be_one != other.can_be_one) || (can_be_zero != other.can_be_zero);
  }

  /// @brief It's uglib but get a string of bits for (T)op, (B)ottom, 0, 1.
  /// @param os
  void print(llvm::raw_ostream &os) const {
    bool one, zero;
    for (int i = 31; i >= 0; --i) {
      one = (can_be_one >> i) & 1;
      zero = (can_be_zero >> i) & 1;
      os << (one ? (zero ? "T" : "1") : (zero ? "0" : "B"));
    };
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const KnownBitsState &state) {
  state.print(os);
  return os;
}

} // namespace known_bits

#endif
