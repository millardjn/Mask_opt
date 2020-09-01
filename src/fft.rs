use ndarray::{s, Array2, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis, Zip};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;

pub fn fft2(input: Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let mut output = Array2::zeros([input.shape()[0], input.shape()[1]]);
    _fft2(input, output.view_mut(), false);
    output
}
pub fn ifft2(input: Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let mut output = Array2::zeros([input.shape()[0], input.shape()[1]]);
    _fft2(input, output.view_mut(), true);
    output
}
pub fn _fft2(
    mut input: Array2<Complex<f64>>,
    mut output: ArrayViewMut2<Complex<f64>>,
    inverse: bool,
) {
    let mut planner = FFTplanner::new(inverse);
    let fft_row = planner.plan_fft(input.shape()[1]);
    let fft_col = planner.plan_fft(input.shape()[0]);
    let normalisation = 1.0 / ((input.shape()[0] * input.shape()[1]) as f64).sqrt();

    Zip::from(output.genrows_mut())
        .and(input.genrows_mut())
        .par_apply(|mut outrow, mut inrow| {
            fft_row.process(
                &mut inrow.as_slice_mut().unwrap(),
                &mut outrow.as_slice_mut().unwrap(),
            );
            //inrow.assign(&outrow)
        });

    Zip::from(output.gencolumns_mut()).par_apply(|mut outcol| {
        let mut vec = outcol.to_vec();
        let mut out = vec![Zero::zero(); outcol.len()];
        fft_col.process(&mut vec, &mut out);
        for (i, e) in outcol.iter_mut().enumerate() {
            *e = out[i] * normalisation;
        }
    });
}

pub fn fft2_shift_inplace(mut input: ArrayViewMut2<Complex<f64>>) {
    for row in input.lanes_mut(Axis(1)) {
        fft_shift_inplace(row);
    }
    for col in input.lanes_mut(Axis(0)) {
        fft_shift_inplace(col);
    }
}

pub fn ifft2_shift_inplace(mut input: ArrayViewMut2<Complex<f64>>) {
    for row in input.lanes_mut(Axis(1)) {
        ifft_shift_inplace(row);
    }
    for col in input.lanes_mut(Axis(0)) {
        ifft_shift_inplace(col);
    }
}

pub fn fft_shift_inplace(mut input: ArrayViewMut1<Complex<f64>>) {
    if input.len() % 2 == 0 {
        return fft_shift_even(input);
    }

    let len = input.len();
    let half = len / 2;

    let mut i = input.len();
    let mut j = half;
    let mut temp1 = input[half];
    for _ in 0..half {
        i -= 1;
        j -= 1;
        let temp2 = temp1;
        temp1 = input[i];
        input[i] = temp2;

        let temp2 = temp1;
        temp1 = input[j];
        input[j] = temp2;
    }
    input[half] = temp1;
}

pub fn ifft_shift_inplace(mut input: ArrayViewMut1<Complex<f64>>) {
    if input.len() % 2 == 0 {
        return fft_shift_even(input);
    }

    let len = input.len();
    let half = len / 2;

    let mut i = 0;
    let mut j = half + 1;
    let mut temp1 = input[half];
    for _ in 0..half {
        let temp2 = temp1;
        temp1 = input[i];
        input[i] = temp2;

        let temp2 = temp1;
        temp1 = input[j];
        input[j] = temp2;

        i += 1;
        j += 1;
    }
    input[half] = temp1;
}

fn fft_shift_even(mut input: ArrayViewMut1<Complex<f64>>) {
    let half = input.len() / 2;
    for i in 0..half {
        let temp = input[i];
        input[i] = input[i + half];
        input[i + half] = temp;
    }
}

pub fn pad_zero_2D(input: ArrayView2<Complex<f64>>) -> Array2<Complex<f64>> {
    let m0 = input.shape()[0];
    let m1 = input.shape()[1];
    let mut out = Array2::zeros([m0 * 2, m1 * 2]);
    let slice = s![
        (m0 + 1) / 2..m0 + (m0 + 1) / 2,
        (m1 + 1) / 2..m1 + (m1 + 1) / 2
    ];
    let mut view = out.slice_mut(slice);
    view.assign(&input);
    out
}

pub fn depad_2D(input: ArrayView2<Complex<f64>>) -> ArrayView2<Complex<f64>> {
    let m0 = input.shape()[0];
    let m1 = input.shape()[1];
    let slice = s![
        (m0 + 1) / 2..m0 + (m0 + 1) / 2,
        (m1 + 1) / 2..m1 + (m1 + 1) / 2
    ];
    input.slice_move(slice)
}

#[cfg(test)]
mod tests {
    use super::{fft2, fft_shift_inplace, ifft2, ifft_shift_inplace};
    use ndarray::{Array2, ArrayViewMut};
    use rustfft::num_complex::Complex;
    use rustfft::num_traits::Zero;

    fn assert_eq_vecs(a: &[Complex<f64>], b: &[Complex<f64>]) {
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).norm() < 1e-7, "{}", (a - b).norm());
        }
    }

    // #[test]
    // fn test_fft() {
    //     let mut input: Vec<Complex<f64>> = vec![1.,2.,3.,4.,5.,6.,7.,8.,9.].into_iter().map(|x| Complex::new(x, 0.)).collect();
    //     let mut output = vec![Zero::zero(); 9];
    //     fft(&mut input, &mut output);
    //     let expected = [Complex::new(45.0,  0.        ), Complex::new(-4.5, 12.36364839), Complex::new(-4.5,   5.36289117),
    //                     Complex::new(-4.5,  2.59807621), Complex::new(-4.5,  0.79347141), Complex::new(-4.5,  -0.79347141),
    //                     Complex::new(-4.5, -2.59807621), Complex::new(-4.5, -5.36289117), Complex::new(-4.5, -12.36364839)];
    //     assert_eq_vecs(&expected, &output);
    // }

    // #[test]
    // fn test_inverse_fft() {
    //     let mut input: Vec<Complex<f64>> = vec![1.,2.,3.,4.,5.,6.,7.,8.,9.].into_iter().map(|x| Complex::new(x, 0.)).collect();
    //     let expected = input.clone();
    //     let mut output = vec![Zero::zero(); 9];
    //     fft(&mut input, &mut output);
    //     let mut output2 = vec![Zero::zero(); 9];
    //     ifft(&mut output, &mut output2);
    //     assert_eq_vecs(&expected, &output2);
    // }

    #[test]
    fn test_fft_shift_odd() {
        let mut input: Vec<Complex<f64>> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();
        let expected: Vec<Complex<f64>> = vec![6., 7., 8., 9., 1., 2., 3., 4., 5.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();

        let input_view = ArrayViewMut::from_shape(9, &mut input).unwrap();
        fft_shift_inplace(input_view);

        assert_eq!(input, expected);
    }

    #[test]
    fn test_fft_shift_even() {
        let mut input: Vec<Complex<f64>> = vec![1., 2., 3., 4., 5., 6., 7., 8.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();
        let expected: Vec<Complex<f64>> = vec![5., 6., 7., 8., 1., 2., 3., 4.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();

        let input_view = ArrayViewMut::from_shape(8, &mut input).unwrap();
        fft_shift_inplace(input_view);

        assert_eq!(input, expected);
    }

    #[test]
    fn test_ifft_shift_odd() {
        let mut input: Vec<Complex<f64>> = vec![6., 7., 8., 9., 1., 2., 3., 4., 5.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();
        let expected: Vec<Complex<f64>> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();

        let input_view = ArrayViewMut::from_shape(9, &mut input).unwrap();
        ifft_shift_inplace(input_view);

        assert_eq!(input, expected);
    }

    #[test]
    fn test_ifft_shift_even() {
        let mut input: Vec<Complex<f64>> = vec![5., 6., 7., 8., 1., 2., 3., 4.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();
        let expected: Vec<Complex<f64>> = vec![1., 2., 3., 4., 5., 6., 7., 8.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();

        let input_view = ArrayViewMut::from_shape(8, &mut input).unwrap();
        ifft_shift_inplace(input_view);

        assert_eq!(input, expected);
    }

    #[test]
    fn test_fft2() {
        let mut input: Vec<Complex<f64>> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();
        let input_view = ArrayViewMut::from_shape((3, 3), &mut input).unwrap();

        let output = fft2(input_view.to_owned());
        

        let expected = [
            Complex::new(15.0, 0.),
            Complex::new(-1.5, 0.86602540333333333333333333333333),
            Complex::new(-1.5, -0.86602540333333333333333333333333),
            Complex::new(-4.5, 2.59807621),
            Complex::new(0.0, 0.),
            Complex::new(0.0, 0.),
            Complex::new(-4.5, -2.59807621),
            Complex::new(0.0, 0.),
            Complex::new(0.0, 0.),
        ];
        assert_eq_vecs(&expected, &output.as_slice().unwrap());
    }

    #[test]
    fn test_inverse_fft2() {
        let mut input: Vec<Complex<f64>> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();
        let input_view = ArrayViewMut::from_shape((3, 3), &mut input).unwrap();

        let output = fft2(input_view.to_owned());

        let output2 = ifft2(output);
        
        let expected: Vec<Complex<f64>> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();
        assert_eq_vecs(&expected, &output2.as_slice().unwrap());
    }
}
