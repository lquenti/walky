//! This module contains 3 constants,
//! used for algorithms to distinguish between sqeuential, parallel, or MPI-based computation.
//! Ideally, these constants should belong into one Enum,
//! but at the moment rust does not support const generics for Enums.

/// use single-threaded computation
pub const SEQ_COMPUTATION: usize = 0;

/// use multi-threaded computation, on one machine
pub const PAR_COMPUTATION: usize = 1;

/// use multiple machinges to parallelize computation, via MPI
#[cfg(feature = "mpi")]
pub const MPI_COMPUTATION: usize = 2;

#[inline]
pub fn panic_on_invaid_mode<const MODE: usize>() -> ! {
    panic!("The Mode {} is not valid", MODE)
}
