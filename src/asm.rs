#![allow(non_snake_case)]

use crate::fft::{fft2, fft2_shift_inplace, ifft2, ifft2_shift_inplace, pad_zero_2D};
use image::{open, ImageBuffer, Luma};
use ndarray::{Array2, ShapeBuilder, Zip};
use num_complex::Complex;
use std::cmp::max;
use std::f64::consts::PI;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use structopt::StructOpt;

use crate::{InputAmplitude, OutputAmplitude};

enum AsmType {
    T1,
    T2,
    T3,
    T4,
}

fn asm_conditions(z: f64, f: f64, m: f64, pitch: f64, lambda: f64) -> AsmType {
    let a = m * pitch * pitch / (lambda * z);
    let b = m * pitch * pitch / (lambda * f);

    if 1.0 > b {
        if 1.0 < a {
            AsmType::T1
        } else {
            AsmType::T2
        }
    } else {
        if 1.0 < a {
            AsmType::T3
        } else {
            AsmType::T4
        }
    }
}

pub (crate) fn asm(
    input: &Array2<Complex<f64>>,
    z: f64,
    f: f64,
    pitch: f64,
    lambda: f64,
    gamma: f64,
) -> OutputAmplitude {
    let m = max(input.shape()[0], input.shape()[1]) as f64;
    match asm_conditions(z, f, m, pitch, lambda) {
        AsmType::T1 => asm_type_1(input, z, f, pitch, lambda, gamma),
        AsmType::T2 => asm_type_2(input, z, f, pitch, lambda, gamma),
        AsmType::T3 => asm_type_3(input, z, f, pitch, lambda, gamma),
        AsmType::T4 => asm_type_4(input, z, f, pitch, lambda, gamma),
    }
}

fn asm_type_1(
    A_xi: &Array2<Complex<f64>>,
    z: f64,
    f: f64,
    pitch: f64,
    lambda: f64,
    gamma: f64,
) -> OutputAmplitude {
    let M = max(A_xi.shape()[0], A_xi.shape()[1]);
    let A_xi_mid = M / 2;
    let freq_resolution = M as f64 / pitch;

    // A(ξ)*f^2/(f^2+ξ^2)*exp(-2πi(sqrt(f^2+ξ^2)-f)/λ)
    let mut intermediate1 = Array2::from_shape_fn([M, M], |(y, x)| {
        let xi0 = (y as f64 - A_xi_mid as f64) * pitch;
        let xi1 = (x as f64 - A_xi_mid as f64) * pitch;

        A_xi[(y, x)]
            * Complex::new(f * f / (xi0 * xi0 + xi1 * xi1 + f * f), 0.0)
            * Complex::new(
                0.0,
                -2.0 * PI / lambda * ((xi0 * xi0 + xi1 * xi1 + f * f).sqrt() - f),
            )
            .exp()
    });
    

    // a(u)
    ifft2_shift_inplace(intermediate1.view_mut());
    let mut a_u = fft2(intermediate1);
    fft2_shift_inplace(a_u.view_mut());

    a_u.indexed_iter_mut().for_each(|((y, x), e)| {
        let u0 = (y as f64 - M as f64) * freq_resolution;
        let u1 = (x as f64 - M as f64) * freq_resolution;

        *e = *e
            * Complex::new(
                0.0,
                2.0 * PI * z * (1.0 / (lambda * lambda) - u0 * u0 - u1 * u1).sqrt(),
            )
            .exp()
    });

    //A(x)
    if gamma == 1.0 {
        ifft2_shift_inplace(a_u.view_mut());
        let mut A_x = ifft2(a_u);
        fft2_shift_inplace(A_x.view_mut());

        OutputAmplitude {
            amplitude: A_x,
            z_position: z,
            resolution: pitch,
        }
    } else if gamma < 1.0 {
        panic!()
    } else {
        let a_u = pad_zero_2D(a_u.view());
        let A_x = scaling_czt(a_u, M, pitch, lambda);
        OutputAmplitude {
            amplitude: A_x,
            z_position: z,
            resolution: pitch/lambda,
        }
    }

}

fn asm_type_2(
    A_xi: &Array2<Complex<f64>>,
    z: f64,
    f: f64,
    pitch: f64,
    lambda: f64,
    gamma: f64,
) -> OutputAmplitude {
    let M = max(A_xi.shape()[0], A_xi.shape()[1]);
    let A_xi_mid = M / 2;

    // A(ξ)*f^2/(f^2+ξ^2)*exp(-2πi(sqrt(f^2+ξ^2)-f)/λ)
    let intermediate1 = Array2::from_shape_fn([M, M], |(y, x)| {
        let xi0 = (y as f64 - A_xi_mid as f64) * pitch;
        let xi1 = (x as f64 - A_xi_mid as f64) * pitch;

        A_xi[(y, x)]
            * Complex::new(f * f / (xi0 * xi0 + xi1 * xi1 + f * f), 0.0)
            * Complex::new(
                0.0,
                -2.0 * PI / lambda * ((xi0 * xi0 + xi1 * xi1 + f * f).sqrt() - f),
            )
            .exp()
    });
    let mut intermediate2 = pad_zero_2D(intermediate1.view());

    // a(u)
    ifft2_shift_inplace(intermediate2.view_mut());
    let mut a_u = fft2(intermediate2);
    fft2_shift_inplace(a_u.view_mut());


    // (z/sqrt(ξ∙ξ+z^2).exp(2πif.sqrt(ξ∙ξ+z^2))).(1/(2π.sqrt(ξ∙ξ+z^2))+i/λ)
    let H_xi = Array2::from_shape_fn([M, M], |(y, x)| {
        let xi0 = (y as f64 - A_xi_mid as f64) * pitch;
        let xi1 = (x as f64 - A_xi_mid as f64) * pitch;

            Complex::new(z /(xi0 * xi0 + xi1 * xi1 + z * z).sqrt(), 0.0)
            * Complex::new(
                0.0,
                2.0 * PI * f * (xi0 * xi0 + xi1 * xi1 + z * z).sqrt(),
            )
            .exp()
            * Complex::new(1.0/(2.0 * PI *(xi0 * xi0 + xi1 * xi1 + z * z).sqrt()), 1.0/lambda)
    });
    let mut H_xi = pad_zero_2D(H_xi.view());

    // h(u)
    ifft2_shift_inplace(H_xi.view_mut());
    let mut h_u = fft2(H_xi);
    fft2_shift_inplace(h_u.view_mut());


    let mut intermediate3 = a_u * h_u;

    //A(x)
    if gamma == 1.0 {
        ifft2_shift_inplace(intermediate3.view_mut());
        let mut A_x = ifft2(intermediate3);
        fft2_shift_inplace(A_x.view_mut());

        OutputAmplitude {
            amplitude: A_x,
            z_position: z,
            resolution: pitch,
        }
    } else if gamma < 1.0 {
        panic!()
    } else {
        let A_x = scaling_czt(intermediate3, M, pitch, lambda);
        OutputAmplitude {
            amplitude: A_x,
            z_position: z,
            resolution: pitch/lambda,
        }
    }
}

fn asm_type_3(
    A_xi: &Array2<Complex<f64>>,
    z: f64,
    f: f64,
    pitch: f64,
    lambda: f64,
    gamma: f64,
) -> OutputAmplitude {

    unimplemented!()
}

fn asm_type_4(
    A_xi: &Array2<Complex<f64>>,
    z: f64,
    f: f64,
    pitch: f64,
    lambda: f64,
    gamma: f64,
) -> OutputAmplitude {

    unimplemented!()
}

// to replace the fixed scale final ifft, a CZT is performed to allow scaling
fn scaling_czt(
    mut F_p_minus_m: Array2<Complex<f64>>,
    M: usize,
    pitch: f64,
    lambda: f64,
) -> Array2<Complex<f64>> {
    assert!(F_p_minus_m.shape() == &[2*M, 2*M]);
    let L = M as f64 * pitch;
    let x0_0 = 0.0;
    let x0_1 = 0.0;

    

    F_p_minus_m.indexed_iter_mut().for_each(|((p0, p1), e)| {
        let p0 = p0 as f64;
        let p1 = p1 as f64;
        let M = M as f64;
        *e = *e
            * Complex::new(
                0.0,
                ((p0 * p0 + p1 * p1) - 2.0 * (M * p0 + M * p1)) * PI / (M * lambda)
                    - ((p0 - M) * x0_0 + (p1 - M) * x0_1) * 2.0 * PI / L,
            )
            .exp();
    });

    let mut intermediate1 = Array2::from_shape_fn([2*M, 2*M], |(p0, p1)| {
        let d0 = p0 as f64 - M as f64;
        let d1 = p1 as f64 - M as f64;
        Complex::new(0.0, -PI*(d0*d0+d1*d1)/(M as f64*lambda)).exp()
    });
    ifft2_shift_inplace(F_p_minus_m.view_mut());
    ifft2_shift_inplace(intermediate1.view_mut());
    let mut out = fft2(ifft2(intermediate1) * ifft2(F_p_minus_m));
    fft2_shift_inplace(out.view_mut());



    out.indexed_iter_mut().for_each(|((m0, m1), e)| {
        let m0 = m0 as f64;
        let m1 = m1 as f64;
        let M = M as f64;

        *e = *e
            * Complex::new(
                0.0,
                PI*(m0*m0+m1*m1 - 2.0 * (M * m0 + M * m1) + 4.0 * M * M)/(M*lambda),
            )
            .exp();
    });

    out
}
