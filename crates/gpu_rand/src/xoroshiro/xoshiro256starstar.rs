use rand_core::impls::fill_bytes_via_next;
use rand_core::le::read_u64_into;
use rand_core::{RngCore, SeedableRng};

/// A xoshiro256** random number generator.
///
/// The xoshiro256** algorithm is not suitable for cryptographic purposes, but
/// is very fast and has excellent statistical properties.
///
/// The algorithm used here is translated from [the `xoshiro256starstar.c`
/// reference source code](http://xoshiro.di.unimi.it/xoshiro256starstar.c) by
/// David Blackman and Sebastiano Vigna.
#[cfg_attr(not(target_os = "cuda"), derive(Copy, cust_core::DeviceCopy))]
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
pub struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    /// Jump forward, equivalently to 2^128 calls to `next_u64()`.
    ///
    /// This can be used to generate 2^128 non-overlapping subsequences for
    /// parallel computations.
    ///
    /// ```
    /// use gpu_rand::xoroshiro::rand_core::SeedableRng;
    /// use gpu_rand::xoroshiro::Xoshiro256StarStar;
    ///
    /// let rng1 = Xoshiro256StarStar::seed_from_u64(0);
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
                0x180ec6d33cfd0aba,
                0xd5a61266f0c9392c,
                0xa9582618e03fc9aa,
                0x39abdc4529b1661c
            ]
        );
    }

    /// Jump forward, equivalently to 2^192 calls to `next_u64()`.
    ///
    /// This can be used to generate 2^64 starting points, from each of which
    /// `jump()` will generate 2^64 non-overlapping subsequences for parallel
    /// distributed computations.
    pub fn long_jump(&mut self) {
        impl_jump!(
            u64,
            self,
            [
                0x76e15d3efefdcbbf,
                0xc5004e441c522fb3,
                0x77710069854ee241,
                0x39109bb02acbe635
            ]
        );
    }

    /// Initializes multiple RNG states such that each state corresponds to a subsequence
    /// separated by `2**128` steps from eachother in the main sequence. This ensures that as long as
    /// no state requests more than `2**128` random numbers, the states are guaranteed to be fully independent.
    #[cfg(not(target_os = "cuda"))]
    pub fn initialize_states(seed: u64, num_states: usize) -> Vec<Self> {
        impl_initialize_states!(seed, num_states)
    }
}

impl SeedableRng for Xoshiro256StarStar {
    type Seed = [u8; 32];

    /// Create a new `Xoshiro256StarStar`.  If `seed` is entirely 0, it will be
    /// mapped to a different seed.
    #[inline]
    fn from_seed(seed: [u8; 32]) -> Xoshiro256StarStar {
        deal_with_zero_seed!(seed, Self);
        let mut state = [0; 4];
        read_u64_into(&seed, &mut state);
        Xoshiro256StarStar { s: state }
    }

    /// Seed a `Xoshiro256StarStar` from a `u64` using `SplitMix64`.
    fn seed_from_u64(seed: u64) -> Xoshiro256StarStar {
        from_splitmix!(seed)
    }
}

impl RngCore for Xoshiro256StarStar {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        // The lowest bits have some linear dependencies, so we use the
        // upper bits instead.
        (self.next_u64() >> 32) as u32
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result_starstar = starstar_u64!(self.s[1]);
        impl_xoshiro_u64!(self);
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
        let mut rng = Xoshiro256StarStar::from_seed([
            1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,
            0, 0, 0,
        ]);
        // These values were produced with the reference implementation:
        // http://xoshiro.di.unimi.it/xoshiro128starstar.c
        let expected = [
            11520,
            0,
            1509978240,
            1215971899390074240,
            1216172134540287360,
            607988272756665600,
            16172922978634559625,
            8476171486693032832,
            10595114339597558777,
            2904607092377533576,
        ];
        for &e in &expected {
            assert_eq!(rng.next_u64(), e);
        }
    }
}
