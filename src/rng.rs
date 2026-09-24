//! The one seeded PRNG in the crate (`splitmix64`), so `queries import --seed` and
//! `queries suggest --seed` are reproducible without an extra dependency.

/// A tiny deterministic PRNG (`splitmix64`).
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    /// A generator whose whole sequence is fixed by `seed`.
    pub(crate) fn new(seed: u64) -> SplitMix64 {
        SplitMix64(seed)
    }

    /// The next 64 random bits.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform float in `[0, 1)`.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A uniform index below `n` (`0` when `n` is `0`).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            usize::try_from(self.next_u64() % (n as u64)).unwrap_or(0)
        }
    }

    /// Shuffle `items` in place (Fisher-Yates).
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_sequence() {
        let mut rng = SplitMix64::new(7);
        let first = rng.next_u64();
        let mut again = SplitMix64::new(7);
        assert_eq!(again.next_u64(), first);
        assert_ne!(again.next_u64(), first);
        let f = SplitMix64::new(1).next_f64();
        assert!((0.0..1.0).contains(&f));
    }

    #[test]
    fn shuffle_is_a_permutation_and_deterministic() {
        let mut items: Vec<u32> = (0..20).collect();
        SplitMix64::new(3).shuffle(&mut items);
        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
        assert_ne!(items, sorted, "twenty items do not shuffle to the identity");
        let mut again: Vec<u32> = (0..20).collect();
        SplitMix64::new(3).shuffle(&mut again);
        assert_eq!(items, again);
        assert_eq!(SplitMix64::new(0).below(0), 0);
        assert!(SplitMix64::new(0).below(5) < 5);
    }
}
