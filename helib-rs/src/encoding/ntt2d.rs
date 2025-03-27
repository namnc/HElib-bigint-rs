// main.rs
use ark_ff::{PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};

pub struct NTTProcessor2D<F: PrimeField> {
    pub n: usize,
    pub n1: usize,
    pub n2: usize,
    pub n_inv: F,
    pub root: F,
    pub pow_tab1: Vec<F>,
    pub pow_tab2: Vec<F>,
    pub inv_pow_tab1: Vec<F>,
    pub inv_pow_tab2: Vec<F>,
    pub ark_ff_domain: Radix2EvaluationDomain<F>,
}

impl<F: PrimeField> NTTProcessor2D<F> {
    /// Constructs a new processor for an n-point transform (n must be a power of 2),
    /// using `root` as a primitive n-th root of unity.
    ///
    /// If `n1` is not provided, it defaults to 2^(k/2), where n = 2^k
    #[inline]
    pub fn new(n: usize, root: F, n1: Option<usize>) -> Self {
        debug_assert!(n.is_power_of_two());

        // Choose n1, defaulting to 2^(k/2)
        let n1 = n1.unwrap_or_else(|| 1 << (n.trailing_zeros() / 2));
        assert!(n % n1 == 0, "N1 must divide N");
        let n2 = n / n1;

        let root1 = root.pow(&[n2 as u64]);
        let root2 = root.pow(&[n1 as u64]);
        let root_inv = root.inverse().expect("mod inverse of root not found");
        let root_inv1 = root1.inverse().expect("mod inverse of root^N2 not found");
        let root_inv2 = root2.inverse().expect("mod inverse of root^N1 not found");
        let n_inv = F::from(n as u64).inverse().expect("inverse not found");

        // Precompute power tables.
        let pow_tab1 = Self::create_pow_table(n1, root1);
        let pow_tab2 = Self::create_pow_table(n2, root2);
        let inv_pow_tab1 = Self::create_pow_table(n1, root_inv1);
        let inv_pow_tab2 = Self::create_pow_table(n2, root_inv2);

        // Create the evaluation domain and set the group generators.
        let mut ark_ff_domain =
            Radix2EvaluationDomain::<F>::new(n).expect("Cannot create ark_ff_domain");
        ark_ff_domain.group_gen = root;
        ark_ff_domain.group_gen_inv = root_inv;

        Self {
            n,
            n1,
            n2,
            n_inv,
            root,
            pow_tab1,
            pow_tab2,
            inv_pow_tab1,
            inv_pow_tab2,
            ark_ff_domain,
        }
    }

    /// Creates a power table of length n/2, computing powers of `root`.
    #[inline]
    fn create_pow_table(n: usize, root: F) -> Vec<F> {
        let half = n >> 1;
        let mut table = Vec::with_capacity(half);
        let mut cur = F::one();
        for _ in 0..half {
            table.push(cur);
            cur *= root;
        }
        table
    }

    /// 2D inverse NTT
    pub fn intt_2d(&self, input: &[F]) -> Vec<F> {
        let mut output = input.to_vec();
        self.intt_2d_inplace(&mut output);
        output
    }

    pub fn intt_2d_inplace(&self, input: &mut [F]) {
        let n1 = self.n1;
        let n2 = self.n2;
        let n = self.n;
        assert_eq!(input.len(), n);

        let mut f = vec![F::zero(); n];
        let mut tmp = vec![F::zero(); n1];

        // Step 1: NTT over n1 (rows by column gathering)
        for j in 0..n2 {
            for i in 0..n1 {
                tmp[i] = input[i * n2 + j];
            }
            Self::ntt_1d_inplace(&mut tmp, &self.inv_pow_tab1);
            for k in 0..n1 {
                f[j * n1 + k] = tmp[k];
            }
        }

        // Step 2: Multiply twiddle factors using repeated powers
        for j in 0..n2 {
            let base = self.root.pow(&[j as u64]).inverse().unwrap();
            let mut tw = F::one();
            for k in 0..n1 {
                f[j * n1 + k] *= tw;
                tw *= base;
            }
        }

        let mut tmp2 = vec![F::zero(); n2];

        // Step 3: NTT over n2
        for k in 0..n1 {
            for j in 0..n2 {
                tmp2[j] = f[j * n1 + k];
            }
            Self::ntt_1d_inplace(&mut tmp2, &self.inv_pow_tab2);
            for j in 0..n2 {
                input[j * n1 + k] = tmp2[j] * self.n_inv;
            }
        }
    }

    pub fn intt_2d_inplace_explicit(&self, input: &mut [F]) {
        // N = total number of elements = N1 * N2
        let N = self.n;
        let inv_N = F::from(N as u64).inverse().unwrap();
        assert_eq!(input.len(), N, "Input length must equal N");

        // Decompose the input as an N1 × N2 matrix (row-major layout)
        let N1 = N / 8;
        let N2 = N / N1;

        // xi is the primitive N-th root of unity.
        // xi_{N1} = xi^N2 for row transforms (n1-axis)
        // xi_{N2} = xi^N1 for column transforms (n2-axis)
        let xi = self.root;
        let xi_N1 = xi.pow(&[N2 as u64]);
        let xi_N2 = xi.pow(&[N1 as u64]);

        let mut output = vec![F::zero(); N];

        // Compute each output coefficient: â_{N1·k2 + k1}
        for k2 in 0..N2 {
            for k1 in 0..N1 {
                let mut acc = F::zero();

                // Loop over n2 (outer dimension)
                for n2 in 0..N2 {
                    let mut inner_sum = F::zero();

                    // Twiddle factor: xi^{–n2·k1}
                    let twiddle = xi.pow(&[(n2 * k1) as u64]).inverse().unwrap();

                    // Loop over n1 (inner dimension)
                    for n1 in 0..N1 {
                        let idx = N2 * n1 + n2; // (n1, n2) flattened
                        let w_n1k1 = xi_N1.pow(&[(n1 * k1) as u64]).inverse().unwrap();
                        inner_sum += input[idx] * w_n1k1;
                    }

                    let w_n2k2 = xi_N2.pow(&[(n2 * k2) as u64]).inverse().unwrap();
                    acc += w_n2k2 * twiddle * inner_sum;
                }

                // Normalize and write to output as â_{N1·k2 + k1}
                output[N1 * k2 + k1] = acc * inv_N;
            }
        }

        // Overwrite input with the computed result
        input.copy_from_slice(&output);
    }

    /// Returns the 2D inverse NTT of the input interpreted as a square matrix (row-major).
    pub fn ntt_2d(&self, input: &[F]) -> Vec<F> {
        let mut output = input.to_vec();
        output = self.ntt_2d_inplace(&mut output);
        output
    }

    pub fn ntt_2d_inplace(&self, hat_a: &[F]) -> Vec<F> {
        let n1 = self.n1;
        let n2 = self.n2;
        assert_eq!(hat_a.len(), n1 * n2, "Invalid input size");

        // Step 1: NTT along columns of hat_a -> f'[k1][n2]
        let mut f_prime = vec![F::zero(); n1 * n2]; // flattened [k1][n2]
        let mut tmp_col = vec![F::zero(); n2];
        for k1 in 0..n1 {
            for k2 in 0..n2 {
                tmp_col[k2] = hat_a[k2 * n1 + k1];
            }
            Self::ntt_1d_inplace(&mut tmp_col, &self.pow_tab2);
            for n2_idx in 0..n2 {
                f_prime[k1 * n2 + n2_idx] = tmp_col[n2_idx];
            }
        }

        // Step 2: f[n2][k1] = xi^{n2.k1} . f'[k1][n2]
        // xi^{n2.k1} = (xi^k1)^n2
        let mut f = vec![F::zero(); n1 * n2]; // flattened [n2][k1]
        let mut powers = vec![F::one(); n1];
        for k1 in 1..n1 {
            powers[k1] = powers[k1 - 1] * self.root;
        }
        for k1 in 0..n1 {
            let mut twiddle = F::one();
            for n2_idx in 0..n2 {
                let val = f_prime[k1 * n2 + n2_idx];
                f[n2_idx * n1 + k1] = val * twiddle;
                twiddle *= powers[k1];
            }
        }

        // Step 3: NTT along rows of f -> result[N2.n1 + n2]
        let mut result = vec![F::zero(); n1 * n2];
        let mut tmp_row = vec![F::zero(); n1];
        for n2_idx in 0..n2 {
            let offset = n2_idx * n1;
            tmp_row.copy_from_slice(&f[offset..offset + n1]);
            Self::ntt_1d_inplace(&mut tmp_row, &self.pow_tab1);
            for n1_idx in 0..n1 {
                result[n1_idx * n2 + n2_idx] = tmp_row[n1_idx];
            }
        }

        result
    }

    /// 1D NTT
    pub fn ntt_1d(slice: &mut [F], pow_table: &[F]) -> Vec<F> {
        let mut output = slice.to_vec();
        Self::ntt_1d_inplace(&mut output, &pow_table);
        output
    }

    /// inplace 1D NTT on an input using a given power table
    fn ntt_1d_inplace(slice: &mut [F], pow_table: &[F]) {
        let m = slice.len();
        let levels = m.ilog2();
        for i in 0..m {
            let j = super::reverse_n_bits(i as u64, levels as u64) as usize;
            if j > i {
                slice.swap(i, j);
            }
        }
        let mut size = 2;
        while size <= m {
            let halfsize = size >> 1;
            let tablestep = m / size;
            for start in (0..m).step_by(size) {
                let mut k = 0;
                for i in start..start + halfsize {
                    let l = i + halfsize;
                    let left = slice[i];
                    let right = slice[l] * pow_table[k];
                    slice[i] = left + right;
                    slice[l] = left - right;
                    k += tablestep;
                }
            }
            size <<= 1;
        }
    }

    //------------------------------------------------
    // arkworks fft wrappers

    /// Returns the FFT of the input.
    pub fn fft(&self, input: &[F]) -> Vec<F> {
        self.ark_ff_domain.fft(input)
    }
    /// inplace FFT.
    pub fn fft_inplace(&self, input: &mut Vec<F>) {
        self.ark_ff_domain.fft_in_place(input)
    }
    /// Returns the inverse FFT of the input.
    pub fn ifft(&self, input: &[F]) -> Vec<F> {
        self.ark_ff_domain.ifft(input)
    }
    /// inplace inverse FFT.
    pub fn ifft_inplace(&self, input: &mut Vec<F>) {
        self.ark_ff_domain.ifft_in_place(input)
    }

    //// Returns the result of negacyclic preprocess on `a`.
    // pub fn negacylcic_preprocess(&self, a: &[F]) -> Vec<F> {
    //     debug_assert_eq!(a.len(), self.n);
    //     let mut tmp = F::one();
    //     let mut a_out = Vec::with_capacity(a.len());
    //     for val in a.iter() {
    //         a_out.push(*val * tmp);
    //         tmp *= self.root;
    //     }
    //     a_out
    // }

    // /// Returns the result of negacyclic postprocess on `a`.
    // pub fn negacylcic_postprocess(&self, a: &[F]) -> Vec<F> {
    //     debug_assert_eq!(a.len(), self.n);
    //     let mut tmp = F::one();
    //     let mut a_out = Vec::with_capacity(a.len());
    //     for val in a.iter() {
    //         a_out.push(*val * tmp);
    //         tmp *= self.root_inverse;
    //     }
    //     a_out
    // }

    // /// Returns the result of negacyclic preprocess on two arrays `a` and `b`.
    // pub fn negacylcic_preprocess_two(&self, a: &[F], b: &[F]) -> (Vec<F>, Vec<F>) {
    //     debug_assert_eq!(a.len(), self.n);
    //     debug_assert_eq!(b.len(), self.n);
    //     let mut tmp = F::one();
    //     let mut a_out = Vec::with_capacity(a.len());
    //     let mut b_out = Vec::with_capacity(b.len());
    //     for (aa, bb) in a.iter().zip(b.iter()) {
    //         a_out.push(*aa * tmp);
    //         b_out.push(*bb * tmp);
    //         tmp *= self.root;
    //     }
    //     (a_out, b_out)
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::galois::Galois;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_poly::Radix2EvaluationDomain;
    use rand::thread_rng;

    // Run several trials.
    const NUM_TRIALS: usize = 10;
    // N must be a perfect square. Here we use 4096 = 64^2.
    const N: usize = 4096;

    #[test]
    fn ntt_2d_random_roundtrip() {
        let root =
            Galois::get_minimal_primitive_n_root_of_unity(N).expect("no root found for NTT!");
        let ntt_proc = NTTProcessor2D::new(N, root, None);
    
        let mut rng = thread_rng();
        for _ in 0..NUM_TRIALS {
            // Fully random input
            let poly: Vec<ark_bn254::Fr> = (0..N).map(|_| ark_bn254::Fr::rand(&mut rng)).collect();
    
            let ntt = ntt_proc.ntt_2d(&poly);
            let recovered = ntt_proc.intt_2d(&ntt);
    
            assert_eq!(
                recovered, poly,
                "2D NTT-INTT roundtrip on random input failed"
            );
        }
    }

    #[test]
    fn ntt_2d_constant_roundtrip() {
        let root =
            Galois::get_minimal_primitive_n_root_of_unity(N).expect("no root found for NTT!");
        let ntt_proc = NTTProcessor2D::new(N, root, None);

        let mut rng = thread_rng();
        for _ in 0..NUM_TRIALS {
            let mu = ark_bn254::Fr::rand(&mut rng);
            let poly = vec![mu; N];

            let ntt = ntt_proc.ntt_2d(&poly);
            let recovered = ntt_proc.intt_2d(&ntt);

            assert_eq!(
                recovered, poly,
                "2D NTT-INTT roundtrip on constant input failed"
            );
        }
    }

    /// Compare 2D NTT with Arkworks' inverse FFT.
    #[test]
    fn test_ark_fft_vs_ntt() {
        let n = N;
        let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        let root = domain.group_gen;
        let proc = NTTProcessor2D::new(n, root, None);
        let mut rng = thread_rng();
        let poly: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let polyclone = poly.clone();

        let ntt_poly = proc.ntt_2d(&polyclone);
        let fft_poly = proc.fft(&poly);
        assert_eq!(ntt_poly, fft_poly, "1D NTT does not match Arkworks' FFT");
    }

    /// Compare 2D inverse NTT with Arkworks' inverse FFT.
    #[test]
    fn test_ark_ifft_vs_intt() {
        let n = N;
        let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        let root = domain.group_gen;
        let proc = NTTProcessor2D::new(n, root, None);
        let mut rng = thread_rng();
        let poly: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

        let intt_poly = proc.intt_2d(&poly);
        let ifft_poly = proc.ifft(&poly);
        assert_eq!(
            intt_poly, ifft_poly,
            "1D inverse NTT does not match Arkworks' IFFT"
        );
    }
}
