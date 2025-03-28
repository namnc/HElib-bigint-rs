use super::{error::Error, CLong};
use ark_ff::PrimeField;
use num_bigint::BigUint;
use std::{ffi::c_void, ptr::null_mut};

#[derive(Debug)]
pub struct ZZ {
    pub(crate) ptr: *mut c_void,
}

impl ZZ {
    pub(crate) fn empty_pointer() -> Self {
        Self { ptr: null_mut() }
    }

    pub fn from_string(mut s: String) -> Result<Self, Error> {
        let mut ptr = null_mut();
        s.push('\0'); // Add null terminator to translate to C string
        let ret = unsafe { helib_bindings::ZZ_from_string(&mut ptr, s.as_ptr() as *const i8) };
        Error::error_from_return(ret)?;
        Ok(Self { ptr })
    }

    pub fn from_long(a: CLong) -> Result<Self, Error> {
        let mut ptr = null_mut();
        let ret = unsafe { helib_bindings::ZZ_from_long(&mut ptr, a) };
        Error::error_from_return(ret)?;
        Ok(Self { ptr })
    }

    pub fn zz_clone(&self) -> Result<ZZ, Error> {
        let mut zz = ZZ::empty_pointer();
        let ret = unsafe { helib_bindings::ZZ_clone(&mut zz.ptr, self.ptr) };
        Error::error_from_return(ret)?;
        Ok(zz)
    }

    pub fn num_bytes(&self) -> Result<CLong, Error> {
        let mut len = 0;
        let ret = unsafe { helib_bindings::ZZ_bytes(self.ptr, &mut len) };
        Error::error_from_return(ret)?;
        Ok(len)
    }

    pub fn is_empty(&self) -> Result<bool, Error> {
        Ok(self.num_bytes()? == 0)
    }

    pub fn len(&self) -> Result<usize, Error> {
        Ok(self.num_bytes()? as usize)
    }

    pub fn from_le_bytes(buf: &[u8]) -> Result<Self, Error> {
        let mut ptr = null_mut();
        let ret =
            unsafe { helib_bindings::ZZ_from_bytes(&mut ptr, buf.as_ptr(), buf.len() as CLong) };
        Error::error_from_return(ret)?;
        Ok(Self { ptr })
    }

    pub fn to_le_bytes(&self) -> Result<Vec<u8>, Error> {
        let len = self.num_bytes()?;
        let mut buf = vec![0u8; len as usize];
        let ret = unsafe { helib_bindings::ZZ_to_bytes(self.ptr, buf.as_mut_ptr(), len) };
        Error::error_from_return(ret)?;
        Ok(buf)
    }

    pub fn destroy(&mut self) -> Result<(), Error> {
        if self.ptr.is_null() {
            return Ok(());
        }

        let ret = unsafe { helib_bindings::ZZ_destroy(self.ptr) };
        Error::error_from_return(ret)?;
        self.ptr = null_mut();
        Ok(())
    }

    pub fn random_mod(mod_: &ZZ) -> Result<Self, Error> {
        let mut ptr = null_mut();
        let ret = unsafe { helib_bindings::ZZ_random(&mut ptr, mod_.ptr) };
        Error::error_from_return(ret)?;
        Ok(Self { ptr })
    }

    pub fn from_biguint(input: BigUint) -> Result<Self, Error> {
        let bytes = input.to_bytes_le();
        Self::from_le_bytes(&bytes)
    }

    pub fn from_fieldelement<F: PrimeField>(input: F) -> Result<Self, Error> {
        Self::from_biguint(input.into())
    }

    pub fn to_biguint(&self) -> Result<BigUint, Error> {
        let bytes = self.to_le_bytes()?;
        Ok(BigUint::from_bytes_le(&bytes))
    }

    pub fn to_fieldelement<F: PrimeField>(&self) -> Result<F, Error> {
        let biguint = self.to_biguint()?;
        Ok(F::from(biguint))
    }

    pub fn char<F: PrimeField>() -> Result<Self, Error> {
        Self::from_biguint(F::MODULUS.into())
    }
}

impl Drop for ZZ {
    fn drop(&mut self) {
        self.destroy().expect("ZZ destroy failed");
    }
}

impl Clone for ZZ {
    fn clone(&self) -> Self {
        self.zz_clone().expect("ZZ clone failed")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_ff::UniformRand;
    use rand::{thread_rng, Rng};

    const TESTRUNS: usize = 10;

    #[test]
    fn from_to_primefield() {
        let mut rng = thread_rng();
        for _ in 0..TESTRUNS {
            let input = ark_bn254::Fr::rand(&mut rng);
            let zz = ZZ::from_fieldelement(input).unwrap();
            let output = zz.to_fieldelement().unwrap();
            assert_eq!(input, output);
        }
    }

    #[test]
    fn zz_from_long() {
        let mut rng = thread_rng();
        for _ in 0..TESTRUNS {
            let mut input = rng.gen::<CLong>();
            let mut zz = ZZ::from_long(input).unwrap();
            let mut zz_bytes = zz.to_le_bytes().unwrap();
            if input < 0 {
                input = -input;
            }
            zz_bytes.resize(size_of::<CLong>(), 0);
            assert_eq!(&zz_bytes, input.to_le_bytes().as_ref());
            zz.destroy().unwrap(); // Is also called in drop
        }
    }

    #[test]
    fn zz_from_string() {
        let mut rng = thread_rng();
        for _ in 0..TESTRUNS {
            let input = ark_bn254::Fr::rand(&mut rng);
            let zz = ZZ::from_string(input.to_string()).unwrap();
            let output = zz.to_fieldelement().unwrap();
            assert_eq!(input, output);
        }
    }

    #[test]
    fn zz_random() {
        let mod_ = ZZ::char::<ark_bn254::Fr>().unwrap();
        let mut prev = ZZ::random_mod(&mod_).unwrap().to_biguint().unwrap();
        for _ in 0..TESTRUNS {
            let zz = ZZ::random_mod(&mod_).unwrap();
            let output = zz.to_biguint().unwrap();
            assert!(output < ark_bn254::Fr::MODULUS.into());
            assert_ne!(prev, output); // Very unlikely to be equal
            prev = output;
        }
    }

    #[test]
    fn zz_clone() {
        let mod_ = ZZ::char::<ark_bn254::Fr>().unwrap();
        for _ in 0..TESTRUNS {
            let mut zz = ZZ::random_mod(&mod_).unwrap();
            let zz_ = zz.to_biguint().unwrap();
            let clone = zz.clone();
            zz.destroy().unwrap();
            let clone_ = clone.to_biguint().unwrap();
            assert_eq!(zz_, clone_);
        }
    }
}
