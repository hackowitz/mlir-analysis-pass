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

#include "llvm/ADT/APInt.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/raw_ostream.h"

namespace known_bits {

struct KnownBitsState {

public:
  unsigned int nbits = 128; // size of long long
  unsigned long long zeroes = 0;
  unsigned long long ones = 0;

  /// @brief Create a known bits state (initially "bottom", all impossible) of the desired size.
  /// @param nbits The number of bits in the underlying data type.
  KnownBitsState(
      unsigned int nbits = 128, unsigned long long zeroes = 0, unsigned long long ones = 0
  )
      : nbits(nbits), zeroes(zeroes), ones(ones) {}
  static KnownBitsState top() { return KnownBitsState(~0, ~0); }
  static KnownBitsState bottom() { return KnownBitsState(0, 0); }
  static KnownBitsState fromConstant(llvm::APInt value) {
    return KnownBitsState(value.getBitWidth(), ~value.getZExtValue(), value.getZExtValue());
  }

  /// @brief A 'join' is a logical or. We start from an impossible "bottom" state and can only move
  /// upwards through the lattice towards a real possible value, the LFP
  static KnownBitsState join(const KnownBitsState &lhs, const KnownBitsState &rhs) {
    return KnownBitsState(
        lhs.nbits > rhs.nbits ? lhs.nbits : rhs.nbits,
        (lhs.zeroes & lhs.nbits) | (rhs.zeroes & rhs.nbits),
        (lhs.ones & lhs.nbits) | (rhs.ones & rhs.nbits)
    );
  }

  /// @brief Get a mask of which bits are actually considered
  unsigned long long mask() const { return (1 << nbits) - 1; }
  unsigned long long tops() const { return zeroes & ones & mask(); }
  unsigned long long bottoms() const { return ~zeroes & ~ones & mask(); }

  /// @brief The minimum is bits that can only be 1 and not 0
  unsigned long long minPossible() const { return ones & ~zeroes; }

  /// @brief The maximum is all bits that can be 1
  unsigned long long maxPossible() const { return ones; }

  /// @brief It's ugly but get a string of bits for top, bottom, 0, and 1.
  void print(llvm::raw_ostream &os) const {
    bool one, zero;
    for (unsigned int i = nbits; i > 0; --i) {
      one = (ones >> (i - 1)) & 1;
      zero = (zeroes >> (i - 1)) & 1;
      os << (one ? (zero ? '?' : '1') : (zero ? '0' : '!'));
    };
  }

  bool operator==(const KnownBitsState &other) const {
    auto m = mask() & other.mask();
    bool eq = ((zeroes & m) == (other.zeroes & m)) && ((ones & m) == (other.ones & m));
    return eq;
  }

  /// @brief Abstrat operator - bits are 0 if either side is 0 and 1 if both sides are 1
  KnownBitsState operator&(const KnownBitsState &other) const {
    // assert(nbits == other.nbits && "Bit count must match");
    return KnownBitsState(nbits, zeroes | other.zeroes, ones & other.ones);
  }

  /// @brief Abstrat operator - bits are 1 if either side is 1 and 0 if both sides are 0
  KnownBitsState operator|(const KnownBitsState &other) const {
    // assert(nbits == other.nbits && "Bit count must match");
    return KnownBitsState(nbits, zeroes & other.zeroes, ones | other.ones);
  }

  /// @brief Abstract operator - bits are 0 if sides are the same and 1 if they are different
  KnownBitsState operator^(const KnownBitsState &other) const {
    // assert(nbits == other.nbits && "Bit count must match");
    return KnownBitsState(
        nbits,
        (zeroes & other.zeroes) | (ones & other.ones),
        (ones & other.zeroes) | (zeroes & other.ones)
    );
  }

  /// @brief Abstract operator - shift left by some number of bits
  KnownBitsState operator<<(const KnownBitsState &other) const {
    // shift left by constant
    if (!other.tops()) {
      // if the other is known, it's value is just it's ones
      return KnownBitsState(nbits, zeroes << other.ones, ones << other.ones);
    }
    // TODO: The easy answer is just to not know, but we can refine the LSBs a tad
    return KnownBitsState(nbits, ~0, ~0 << other.minPossible());
  }

  /// @brief Abstract operator - shift right by some number of bits
  KnownBitsState operator>>(const KnownBitsState &other) const {
    // shift right by constant
    if (!other.tops()) {
      // if the other is known, it's value is just it's ones
      return KnownBitsState(nbits, zeroes >> other.ones, ones >> other.ones);
    }
    // TODO: The easy answer is just to not know, but we can refine the LSBs a tad
    return KnownBitsState(nbits, ~0, ~0 >> other.minPossible());
  }

  KnownBitsState zeroExtend(unsigned int width) const {
    return KnownBitsState(width, zeroes & mask(), ones & mask());
  }

  /// @brief Very ugly way to sign extend manually. I sure hope the complier makes this good
  KnownBitsState signExtend(unsigned int width) const {
    KnownBitsState other = zeroExtend(width);
    unsigned long long msb = 1 << (nbits - 1);
    unsigned long long ext_mask = other.mask() & ~mask();
    if (zeroes & msb)
      other.zeroes |= ext_mask;
    if (ones & msb)
      other.ones |= ext_mask;
    return other;
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const KnownBitsState &state) {
  state.print(os);
  return os;
}

} // namespace known_bits

#endif
