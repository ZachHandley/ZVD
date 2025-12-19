//! Rational number representation for timestamps and frame rates

use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

/// A rational number represented as numerator/denominator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational {
    pub num: i64,
    pub den: i64,
}

impl Rational {
    /// Create a new rational number
    pub fn new(num: i64, den: i64) -> Self {
        let mut r = Rational { num, den };
        r.reduce();
        r
    }

    /// Create a rational from an integer
    pub fn from_int(n: i64) -> Self {
        Rational { num: n, den: 1 }
    }

    /// Create a rational from a float (approximate)
    pub fn from_float(f: f64) -> Self {
        const PRECISION: i64 = 1_000_000;
        let num = (f * PRECISION as f64).round() as i64;
        Rational::new(num, PRECISION)
    }

    /// Convert to floating point
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Reduce the fraction to lowest terms
    fn reduce(&mut self) {
        if self.den == 0 {
            return;
        }

        let gcd = Self::gcd(self.num.abs(), self.den.abs());
        if gcd > 1 {
            self.num /= gcd;
            self.den /= gcd;
        }

        // Keep denominator positive
        if self.den < 0 {
            self.num = -self.num;
            self.den = -self.den;
        }
    }

    /// Calculate greatest common divisor
    fn gcd(mut a: i64, mut b: i64) -> i64 {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    }

    /// Invert the rational number
    pub fn invert(self) -> Self {
        Rational::new(self.den, self.num)
    }
}

impl Default for Rational {
    fn default() -> Self {
        Rational { num: 0, den: 1 }
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.num, self.den)
    }
}

impl Add for Rational {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Rational::new(
            self.num * other.den + other.num * self.den,
            self.den * other.den,
        )
    }
}

impl Sub for Rational {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Rational::new(
            self.num * other.den - other.num * self.den,
            self.den * other.den,
        )
    }
}

impl Mul for Rational {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Rational::new(self.num * other.num, self.den * other.den)
    }
}

impl Div for Rational {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Rational::new(self.num * other.den, self.den * other.num)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_creation() {
        let r = Rational::new(1, 2);
        assert_eq!(r.num, 1);
        assert_eq!(r.den, 2);
    }

    #[test]
    fn test_rational_reduction() {
        let r = Rational::new(2, 4);
        assert_eq!(r.num, 1);
        assert_eq!(r.den, 2);
    }

    #[test]
    fn test_rational_arithmetic() {
        let a = Rational::new(1, 2);
        let b = Rational::new(1, 3);

        let sum = a + b;
        assert_eq!(sum.num, 5);
        assert_eq!(sum.den, 6);

        let diff = a - b;
        assert_eq!(diff.num, 1);
        assert_eq!(diff.den, 6);
    }

    #[test]
    fn test_rational_to_float() {
        let r = Rational::new(1, 2);
        assert_eq!(r.to_f64(), 0.5);
    }
}
