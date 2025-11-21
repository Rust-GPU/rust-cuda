use rand_core::impls::fill_bytes_via_next;
use rand_core::le::read_u64_into;
use rand_core::{RngCore, SeedableRng};

/// A xoroshiro128+ random number generator.
///
/// The xoroshiro128+ algorithm is not suitable for cryptographic purposes, but
/// is very fast and has good statistical properties, besides a low linear
/// complexity in the lowest bits.
///
/// The algorithm used here is translated from [the `xoroshiro128plus.c`
/// reference source code](http://xoshiro.di.unimi.it/xoroshiro128plus.c) by
/// David Blackman and Sebastiano Vigna.
#[allow(missing_copy_implementations)]
#[cfg_attr(not(target_os = "cuda"), derive(Copy, cust_core::DeviceCopy))]
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
pub struct Xoroshiro128Plus {
    s0: u64,
    s1: u64,
}

impl Xoroshiro128Plus {
    /// Jump forward, equivalently to 2^64 calls to `next_u64()`.
    ///
    /// This can be used to generate 2^64 non-overlapping subsequences for
    /// parallel computations.
    ///
    /// ```
    /// use gpu_rand::xoroshiro::rand_core::SeedableRng;
    /// use gpu_rand::xoroshiro::Xoroshiro128Plus;
    ///
    /// let rng1 = Xoroshiro128Plus::seed_from_u64(0);
    /// let mut rng2 = rng1.clone();
    /// rng2.jump();
    /// let mut rng3 = rng2.clone();
    /// rng3.jump();
    /// ```
    pub fn jump(&mut self) {
        impl_jump!(u64, self, [0xdf900294d8f554a5, 0x170865df4b3201fc]);
    }

    /// Jump forward, equivalently to 2^96 calls to `next_u64()`.
    ///
    /// This can be used to generate 2^32 starting points, from each of which
    /// `jump()` will generate 2^32 non-overlapping subsequences for parallel
    /// distributed computations.
    pub fn long_jump(&mut self) {
        impl_jump!(u64, self, [0xd2a98b26625eee7b, 0xdddf9b1090aa7ac1]);
    }

    /// Initializes multiple RNG states such that each state corresponds to a subsequence
    /// separated by `2**64` steps from eachother in the main sequence. This ensures that as long as
    /// no state requests more than `2**64` random numbers, the states are guaranteed to be fully independent.
    #[cfg(not(target_os = "cuda"))]
    pub fn initialize_states(seed: u64, num_states: usize) -> Vec<Self> {
        impl_initialize_states!(seed, num_states)
    }
}

impl RngCore for Xoroshiro128Plus {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        // The two lowest bits have some linear dependencies, so we use the
        // upper bits instead.
        (self.next_u64() >> 32) as u32
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let r = self.s0.wrapping_add(self.s1);
        impl_xoroshiro_u64!(self);
        r
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        fill_bytes_via_next(self, dest);
    }
}

impl SeedableRng for Xoroshiro128Plus {
    type Seed = [u8; 16];

    /// Create a new `Xoroshiro128Plus`.  If `seed` is entirely 0, it will be
    /// mapped to a different seed.
    fn from_seed(seed: [u8; 16]) -> Xoroshiro128Plus {
        deal_with_zero_seed!(seed, Self);
        let mut s = [0; 2];
        read_u64_into(&seed, &mut s);

        Xoroshiro128Plus { s0: s[0], s1: s[1] }
    }

    /// Seed a `Xoroshiro128Plus` from a `u64` using `SplitMix64`.
    fn seed_from_u64(seed: u64) -> Xoroshiro128Plus {
        from_splitmix!(seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference() {
        let mut rng = Xoroshiro128Plus::from_seed([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
        // These values were produced with the reference implementation:
        // http://xoshiro.di.unimi.it/xoshiro128starstar.c
        let expected = [
            3,
            412333834243,
            2360170716294286339,
            9295852285959843169,
            2797080929874688578,
            6019711933173041966,
            3076529664176959358,
            3521761819100106140,
            7493067640054542992,
            920801338098114767,
        ];
        for &e in &expected {
            assert_eq!(rng.next_u64(), e);
        }
    }
}
