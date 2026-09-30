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
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/raw_ostream.h"

// I've never written macros before but they seem simple enough
// MAX gets used in binary comparisons but might produce tops when it should bottoms
// FIXME: ...or worse, fail to sign-extend the smaller value and be incorrect analysis
#define MASK(n) ((n) < ULLONG_WIDTH ? (1ULL << (n)) - 1ULL : ~0ULL)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

using ull = unsigned long long;

namespace known_bits {

struct KnownBitsState {

public:
  ull nbits = ULLONG_WIDTH; // size of ull
  ull zeroes = 0;
  ull ones = 0;

  /// @brief Create a known bits state (initially "bottom", all impossible) of the desired size.
  /// @param nbits The number of bits in the underlying data type.
  KnownBitsState(ull nbits = ULLONG_WIDTH, ull zeroes = 0ULL, ull ones = 0ULL)
      : nbits(nbits), zeroes(zeroes), ones(ones) {}

  /// @brief Top is {0, 1} for the bit width, and {} for bits out of range.
  static KnownBitsState top(ull nbits = ULLONG_WIDTH) {
    return KnownBitsState(nbits, MASK(nbits), MASK(nbits));
  }

  /// @brief Bottom is {} for all bits, in or out of `nbits` range.
  static KnownBitsState bottom(ull nbits = ULLONG_WIDTH) {
    return KnownBitsState(nbits, 0ULL, 0ULL);
  }

  static KnownBitsState zero(ull nbits = ULONG_WIDTH) {
    return KnownBitsState(nbits, ~0ULL & MASK(nbits), 0ULL);
  }

  /// @brief All bits are known up to the bit width, and bottom above the bit width
  static KnownBitsState fromConstant(llvm::APInt value) {
    ull n = value.getBitWidth();
    ull v = value.getZExtValue();
    return KnownBitsState(n, ~v, v);
    // ull m = MASK(n); // I don't know why, but using the mask ruins a few things
    // return KnownBitsState(n, ((~v) & m), (v & m)); // never too many parentheses
  }

  /// @brief A 'join' is a logical or. We start from an impossible "bottom" state and can only move
  /// upwards through the lattice towards a real possible value, the LFP
  /// This is commutative because `&`, `|`, and "select max" are all commutative
  /// This is idempotenet because `x | y | y`, `x & y & y`, and "select max" are idempotent
  ///
  /// @note 'bottom' is handled implicitly bitwise since 0 | x == x | 0 == x.
  /// The nbits is the minimum since 'bottom' has the maximum.
  /// I don't know all the implications of this choice.
  static KnownBitsState join(const KnownBitsState &lhs, const KnownBitsState &rhs) {
    auto l0 = lhs.zeroes & lhs.mask();
    auto r0 = rhs.zeroes & rhs.mask();
    auto l1 = lhs.ones & lhs.mask();
    auto r1 = rhs.ones & rhs.mask();
    KnownBitsState state(MIN(lhs.nbits, rhs.nbits), l0 | r0, l1 | r1);
    return state;
  }

  /// @brief Get a mask of which bits are actually used
  ull mask() const { return MASK(nbits); }
  ull tops() const { return (zeroes & ones) & mask(); }
  ull bottoms() const { return ~(zeroes | ones) & mask(); }
  bool isBottom() const { return !((zeroes | ones) & mask()); }
  bool isTop() const { return (zeroes & ones & mask()) == mask(); }
  bool isInteresting() const { return (zeroes ^ ones) & mask(); }
  bool isConstant() const { return (zeroes ^ ones) == mask(); }

  /// @brief The minimum is bits that can only be 1 and not 0
  ull minPossible() const { return ones & ~zeroes; }

  /// @brief The maximum is all bits that can be 1
  ull maxPossible() const { return ones; }

  /// @brief It's ugly but get a string of bits for top, bottom, 0, and 1.
  void print(llvm::raw_ostream &os) const {
    bool one, zero;
    os << nbits << "'b";        // verilog style
    for (ull i = nbits; i--;) { // post-decrement loops (nbits, 0]
      one = (ones >> i) & 1;
      zero = (zeroes >> i) & 1;
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
    return KnownBitsState(MAX(nbits, other.nbits), zeroes | other.zeroes, ones & other.ones);
  }

  /// @brief Abstrat operator - bits are 1 if either side is 1 and 0 if both sides are 0
  KnownBitsState operator|(const KnownBitsState &other) const {
    return KnownBitsState(MAX(nbits, other.nbits), zeroes & other.zeroes, ones | other.ones);
  }

  /// @brief Abstract operator - bits are 0 if sides are the same and 1 if they are different
  KnownBitsState operator^(const KnownBitsState &other) const {
    return KnownBitsState(
        MAX(nbits, other.nbits),
        (zeroes & other.zeroes) | (ones & other.ones),
        (ones & other.zeroes) | (zeroes & other.ones)
    );
  }

  /// @brief Abstract operator - shift left by some number of bits
  KnownBitsState operator<<(const KnownBitsState &other) const {
    // shift left by constant
    if (other.isConstant())
      // if the other is known, it's value is just it's ones
      return KnownBitsState(nbits, zeroes << other.ones, ones << other.ones);

    // TODO: The easy answer is just to not know, but we can refine the LSBs a tad
    return KnownBitsState(nbits, ~0ULL, ~0ULL << other.minPossible());
  }

  /// @brief Abstract operator - shift right by some number of bits
  KnownBitsState operator>>(const KnownBitsState &other) const {
    // shift right by constant
    if (other.isConstant()) {
      // if the other is known, it's value is just it's ones
      return KnownBitsState(nbits, zeroes >> other.ones, ones >> other.ones);
    }
    // TODO: The easy answer is just to not know, but we can refine the LSBs a tad
    return KnownBitsState(nbits, ~0ULL, ~0ULL >> other.minPossible());
  }

  KnownBitsState operator+(const KnownBitsState &other) const {
    ull LHSKnownOnes = ones & ~zeroes;
    ull RHSKnownOnes = other.ones & ~other.zeroes;

    ull LHSKnownZeroes = zeroes & ~ones;
    ull RHSKnownZeroes = other.zeroes & ~other.ones;

    // TODO: known carries can be refined by considering ripple carry
    ull KnownCarryOut = (LHSKnownOnes & RHSKnownOnes);
    ull KnownNotCarryOut = (LHSKnownZeroes & RHSKnownZeroes);
    ull KnownCarryIn = (KnownCarryOut << 1);       // | carry in bit for subtraction
    ull KnownNotCarryIn = (KnownNotCarryOut << 1); // | carry in bit for subtraction

    ull KnownSums = (LHSKnownOnes & RHSKnownZeroes) ^ (RHSKnownOnes & LHSKnownZeroes);
    ull KnownNotSums = (LHSKnownZeroes & RHSKnownZeroes) | KnownCarryOut;

    ull KnownOnes = (KnownNotSums & KnownCarryIn) | (KnownSums & KnownNotCarryIn);
    ull KnownZeroes = (KnownSums & KnownCarryIn) | (KnownNotSums & KnownNotCarryIn);
    ull Unknowns = ~(KnownOnes | KnownZeroes);
    return KnownBitsState(nbits, Unknowns | KnownZeroes, Unknowns | KnownOnes);
  }

  KnownBitsState zeroExtend(ull width) const {
    KnownBitsState state(width, zeroes & mask(), ones & mask());
    state.zeroes |= state.mask() & ~mask();
    return state;
  }

  /// @brief Very ugly way to sign extend manually. I sure hope the complier makes this good
  KnownBitsState signExtend(ull width) const {
    KnownBitsState other = zeroExtend(width);
    ull msb = 1 << (nbits - 1);
    ull ext_mask = other.mask() & ~mask();
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
