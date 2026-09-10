//! Inline shape and strides.
//!
//! Every array result needs a shape and strides. Storing them in a `Vec`
//! costs a heap allocation per result (about 20 ns, a quarter of a
//! small-array add), so they live inline in fixed arrays. Strides are in
//! bytes, as in NumPy, so they can be handed to the buffer protocol without
//! conversion.

use std::fmt;

/// Maximum number of dimensions handled natively. Anything larger is
/// delegated to NumPy.
pub const MAX_NDIM: usize = 8;

/// Size of one element in bytes (float64 only for now).
pub const ITEMSIZE: isize = 8;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Dims {
    ndim: u8,
    shape: [usize; MAX_NDIM],
    /// C-contiguous strides in bytes. Entries beyond `ndim` are zero.
    strides: [isize; MAX_NDIM],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimsError {
    TooManyDims(usize),
}

impl fmt::Display for DimsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimsError::TooManyDims(n) => write!(
                f,
                "lightarray supports at most {MAX_NDIM} dimensions natively, got {n}"
            ),
        }
    }
}

impl Dims {
    /// Contiguous C-order dims for `shape`.
    pub fn from_shape(shape: &[usize]) -> Result<Dims, DimsError> {
        if shape.len() > MAX_NDIM {
            return Err(DimsError::TooManyDims(shape.len()));
        }
        let mut d = Dims { ndim: shape.len() as u8, shape: [0; MAX_NDIM], strides: [0; MAX_NDIM] };
        d.shape[..shape.len()].copy_from_slice(shape);
        let mut stride = ITEMSIZE;
        for i in (0..shape.len()).rev() {
            d.strides[i] = stride;
            stride *= shape[i] as isize;
        }
        Ok(d)
    }

    pub fn scalar() -> Dims {
        Dims { ndim: 0, shape: [0; MAX_NDIM], strides: [0; MAX_NDIM] }
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.ndim as usize
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim as usize]
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides[..self.ndim as usize]
    }

    /// Raw pointers for the buffer protocol; only valid while `self` is alive
    /// and not moved.
    pub fn shape_ptr(&self) -> *const usize {
        self.shape.as_ptr()
    }

    pub fn strides_ptr(&self) -> *const isize {
        self.strides.as_ptr()
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Dims of `self` with the leading axis removed (result of `a[i]`).
    pub fn drop_leading(&self, n: usize) -> Dims {
        Dims::from_shape(&self.shape()[n..]).expect("fewer dims than self")
    }
}

impl fmt::Debug for Dims {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dims").field("shape", &self.shape()).field("strides", &self.strides()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_c_order_bytes() {
        let d = Dims::from_shape(&[2, 3, 4]).unwrap();
        assert_eq!(d.shape(), &[2, 3, 4]);
        assert_eq!(d.strides(), &[96, 32, 8]);
        assert_eq!(d.size(), 24);
        assert_eq!(d.drop_leading(1).shape(), &[3, 4]);
    }

    #[test]
    fn scalar_and_limits() {
        assert_eq!(Dims::scalar().size(), 1);
        assert_eq!(Dims::from_shape(&[]).unwrap(), Dims::scalar());
        assert!(Dims::from_shape(&[1; MAX_NDIM + 1]).is_err());
    }
}
