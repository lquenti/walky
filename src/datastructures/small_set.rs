use core::fmt;

/// `SmallSet` is a set data structure. It my contain integer values from the range
/// `START..START+`[`usize::BITS`] (exclusive).
/// The set is represented via one [`usize`] value.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct SmallSet<const START: usize> {
    data: usize,
}

impl<const START: usize> SmallSet<START> {
    /// creates an empty set.
    pub const fn new() -> Self {
        SmallSet { data: 0 }
    }

    const fn from_bits(bits: usize) -> Self {
        SmallSet { data: bits }
    }

    /// dumps the bit-representation of the set
    pub const fn to_bits(&self) -> usize {
        self.data
    }

    /// Returns the cardinality of the set. Will at most be `usize::BITS`.
    pub const fn len(&self) -> u32 {
        self.data.count_ones()
    }

    /// Checks, if the set is empty.
    pub const fn is_empty(&self) -> bool {
        self.data == 0
    }

    /// creates the singleton `{k}`
    pub const fn singleton(k: usize) -> Self {
        debug_assert!(k >= START, "cannot store this k, it is too small");
        debug_assert!(
            k - START < usize::BITS as usize,
            "cannot store this k, it is too large"
        );
        SmallSet {
            data: 1 << (k - START),
        }
    }

    /// returns the set `S / {k}`
    /// #panics
    /// if `k < START`
    pub const fn remove(&self, k: usize) -> Self {
        assert!(k >= START, "k has to be larger than START");
        let k = k - START;
        Self::from_bits(self.data & !(1 << k))
    }

    /// returns the set `S union {k}`
    /// #panics
    /// if `k < START`
    pub const fn insert(&self, k: usize) -> Self {
        assert!(k >= START, "k has to be larger than START");
        let k = k - START;
        Self::from_bits(self.data | (1 << k))
    }

    /// returns an iterator over all subsets of `{START..=START+n}`
    ///
    /// # panics
    /// if `n > `[`usize::BITS`] or `n < START`
    pub fn enumerate_sets(n: usize) -> impl Iterator<Item = Self> {
        assert!(
            n >= START,
            "n={} has to be at least as big as START={}",
            n,
            START
        );

        let n = n - START;

        assert!(
            n <= usize::BITS as usize,
            "n ({}) has to be less or equal to {}",
            n,
            usize::BITS
        );
        // maxn = 2^{n+1}
        let maxn = 2usize << n;
        (0..maxn).map(|bits| Self::from_bits(bits))
    }

    /// returns an iterator over all items in the set
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..usize::BITS)
            .filter(|k| (self.data >> k) & 1 != 0)
            .map(|idx| START + idx as usize)
    }
}

impl<const START: usize> Default for SmallSet<START> {
    /// yields the empty set
    fn default() -> Self {
        Self::new()
    }
}

impl<const START: usize> fmt::Display for SmallSet<START> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        for k in self.iter() {
            write!(f, "{}, ", k)?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_enumerate_sets() {
        let sets = SmallSet::<1>::enumerate_sets(3).collect::<HashSet<_>>();
        let expected = vec![0b000, 0b001, 0b010, 0b100, 0b011, 0b101, 0b110, 0b111]
            .into_iter()
            .map(|bits| SmallSet::<1>::from_bits(bits))
            .collect();
        assert_eq!(sets, expected);
    }

    #[test]
    fn test_smallset_iter() {
        let set = SmallSet::<1>::from_bits(0b1101);
        let bit_idxs: Vec<_> = set.iter().collect();
        let expected = vec![1, 3, 4];
        assert_eq!(bit_idxs, expected)
    }

    #[test]
    fn test_remove_set_item() {
        println!("{}", 2 << 0);
        let set = SmallSet::<2>::from_bits(0b1101);
        let set_minus_item_4 = set.remove(4);
        let expected = SmallSet::<2>::from_bits(0b1001);
        assert_eq!(set_minus_item_4, expected);
    }

    #[test]
    fn test_display_small_set() {
        let set = SmallSet::<1>::from_bits(0b101);
        assert_eq!(set.to_string(), "{1, 3, }");
    }

    #[test]
    fn test_is_empty_set() {
        let set = SmallSet::<3>::new();
        assert!(set.is_empty());

        let set = set.insert(4);
        assert!(!set.is_empty(), "set should contain the value 4");
    }
}
