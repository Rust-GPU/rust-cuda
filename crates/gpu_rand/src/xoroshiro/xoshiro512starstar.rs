use rand_core::impls::fill_bytes_via_next;
use rand_core::le::read_u64_into;
use rand_core::{RngCore, SeedableRng};

use crate::xoroshiro::Seed512;

/// A xoshiro512** random number generator.
///
/// The xoshiro512** algorithm is not suitable for cryptographic purposes, but
/// is very fast and has excellent statistical properties.
///
/// The algorithm used here is translated from [the `xoshiro512starstar.c`
/// reference source code](http://xoshiro.di.unimi.it/xoshiro512starstar.c) by
/// David Blackman and Sebastiano Vigna.
#[cfg_attr(not(target_os = "cuda"), derive(Copy, cust_core::DeviceCopy))]
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
pub struct Xoshiro512StarStar {
    s: [u64; 8],
}

impl Xoshiro512StarStar {
    /// Jump forward, equivalently to 2^256 calls to `next_u64()`.
    ///
    /// This can be used to generate 2^256 non-overlapping subsequences for
    /// parallel computations.
    ///
    /// ```
    /// use gpu_rand::xoroshiro::rand_core::SeedableRng;
    /// use gpu_rand::xoroshiro::Xoshiro512StarStar;
    ///
    /// let rng1 = Xoshiro512StarStar::seed_from_u64(0);
    /// let mut rng2 = rng1.clone();
    /// rng2.jump();
    /// let mut rng3 = rng2.clone();
    /// rng3.jump();
    /// ```
    pub fn jump(&mut self) {
        impl_jump!(
            u64,
            self,
            [
                0x33ed89b6e7a353f9,
                0x760083d7955323be,
                0x2837f2fbb5f22fae,
                0x4b8c5674d309511c,
                0xb11ac47a7ba28c25,
                0xf1be7667092bcc1c,
                0x53851efdb6df0aaf,
                0x1ebbc8b23eaf25db
            ]
        );
    }

    /// Jump forward, equivalently to 2^384 calls to `next_u64()`.
    ///
    /// This can be used to generate 2^128 starting points, from each of which
    /// `jump()` will generate 2^128 non-overlapping subsequences for parallel
    /// distributed computations.
    pub fn long_jump(&mut self) {
        impl_jump!(
            u64,
            self,
            [
                0x11467fef8f921d28,
                0xa2a819f2e79c8ea8,
                0xa8299fc284b3959a,
                0xb4d347340ca63ee1,
                0x1cb0940bedbff6ce,
                0xd956c5c4fa1f8e17,
                0x915e38fd4eda93bc,
                0x5b3ccdfa5d7daca5
            ]
        );
    }

    /// Initializes multiple RNG states such that each state corresponds to a subsequence
    /// separated by `2**256` steps from eachother in the main sequence. This ensures that as long as
    /// no state requests more than `2**256` random numbers, the states are guaranteed to be fully independent.
    #[cfg(not(target_os = "cuda"))]
    pub fn initialize_states(seed: u64, num_states: usize) -> Vec<Self> {
        impl_initialize_states!(seed, num_states)
    }
}

impl SeedableRng for Xoshiro512StarStar {
    type Seed = Seed512;

    /// Create a new `Xoshiro512StarStar`.  If `seed` is entirely 0, it will be
    /// mapped to a different seed.
    #[inline]
    fn from_seed(seed: Seed512) -> Xoshiro512StarStar {
        deal_with_zero_seed!(seed, Self);
        let mut state = [0; 8];
        read_u64_into(&seed.0, &mut state);
        Xoshiro512StarStar { s: state }
    }

    /// Seed a `Xoshiro512StarStar` from a `u64` using `SplitMix64`.
    fn seed_from_u64(seed: u64) -> Xoshiro512StarStar {
        from_splitmix!(seed)
    }
}

impl RngCore for Xoshiro512StarStar {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        // The lowest bits have some linear dependencies, so we use the
        // upper bits instead.
        (self.next_u64() >> 32) as u32
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result_starstar = starstar_u64!(self.s[1]);
        impl_xoshiro_large!(self);
        result_starstar
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        fill_bytes_via_next(self, dest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference() {
        let mut rng = Xoshiro512StarStar::from_seed(Seed512([
            1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,
            0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 8, 0,
            0, 0, 0, 0, 0, 0,
        ]));
        // These values were produced with the reference implementation:
        // http://xoshiro.di.unimi.it/xoshiro512starstar.c
        let expected = [
            11520,
            0,
            23040,
            23667840,
            144955163520,
            303992986974289920,
            25332796375735680,
            296904390158016,
            13911081092387501979,
            15304787717237593024,
        ];
        for &e in &expected {
            assert_eq!(rng.next_u64(), e);
        }
    }
}
