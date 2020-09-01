use crate::fft::{fft2, fft2_shift_inplace, ifft2_shift_inplace};
use image::{open, ImageBuffer, Luma};
use ndarray::{Array2, ShapeBuilder, Zip};
use num_complex::Complex;
use std::cmp::max;
use std::f64::consts::PI;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use structopt::StructOpt;

mod asm;
mod fft;

#[derive(StructOpt, Debug)]
#[structopt(name = "basic")]
struct Opt {
    /// Output file
    #[structopt(short, long, parse(from_os_str))]
    output: PathBuf,

    #[structopt(short, long, parse(from_os_str))]
    mask: PathBuf,

    #[structopt(short, long)]
    diameter: f64,

    #[structopt(short, long)]
    focal_length: f64,

    #[structopt(short, long)]
    wavelength: f64,
}

fn main() {
    let opt = Opt::from_args();
    let img = open(&opt.mask)
        .unwrap_or_else(|err| panic!("Failed to open {:?} due to {:?}", opt.mask, err))
        .into_luma();
    let mut input = InputAmplitude {
        apeture_diameter: opt.diameter,
        wavelength: opt.wavelength,
        amplitude: Array2::from_shape_fn(
            [img.height() as usize, img.width() as usize].f(),
            |(y, x)| Complex::new(img.get_pixel(x as u32, y as u32).0[0] as f64 / 255.0, 0.0),
        )
        .as_standard_layout()
        .to_owned(),
    };

    input.apply_focal_length_phase(opt.focal_length);

    let airy = airy_radius(opt.wavelength, opt.focal_length, opt.diameter);
    let output = rayleigh_sommerfeld(&input, 600, airy * 0.5, opt.focal_length);

   

    let mut max = 0.0;
    let amp = output.amplitude.map(|e| {
        let amp = e.re * e.re + e.im * e.im;
        if amp > max {
            max = amp;
        }
        amp
    });

    let output_img = ImageBuffer::from_fn(
        output.amplitude.shape()[1] as u32,
        output.amplitude.shape()[0] as u32,
        |x, y| {
            let val = amp[[y as usize, x as usize]] / max;

            //let val = ((val).ln()/8.0 + 1.0).max(0.0).min(1.0);
            let val = (val / 0.004).max(0.0).min(1.0);
            //let val = (val).max(0.0).min(1.0);

            Luma([(val.powf(1.0 / 2.2) * 255.0) as u8])
        },
    );

    output_img
        .save(&opt.output)
        .unwrap_or_else(|err| panic!("Failed to save to {:?} due to {:?}", opt.mask, err));
}

fn circular() {
    let fl = 1.0;
    let apeture = 0.2;
    let wavelength = 500E-9;

    let mut input = InputAmplitude::new_square(wavelength, apeture, 0.001);
    input.apply_circular_apeture();
    input.apply_focal_length_phase(fl);
    let airy = airy_radius(wavelength, fl, apeture);
    let output = rayleigh_sommerfeld(&input, 1000, airy * 0.025, fl);

    println!("{:#?}", output.resolution);
    println!("{:#?}", output.z_position);
    for row in output.amplitude.outer_iter() {
        println!();
        for e in row {
            print!("{} ", e.re * e.re + e.im * e.im);
        }
    }
}

struct InputAmplitude {
    /// D
    apeture_diameter: f64,

    /// λ
    wavelength: f64,
    amplitude: Array2<Complex<f64>>,
}

impl InputAmplitude {
    fn new_square(wavelength: f64, apeture_diameter: f64, resolution: f64) -> Self {
        let size = (apeture_diameter / resolution).ceil() as usize;
        Self {
            apeture_diameter,
            wavelength,
            amplitude: Array2::from_elem([size, size], Complex::new(1.0, 0.0))
                .as_standard_layout()
                .to_owned(),
        }
    }

    fn apply_focal_length_phase(&mut self, focal_length: f64) {
        let z = focal_length;
        let hi = self.amplitude.shape()[0] as f64;
        let wi = self.amplitude.shape()[1] as f64;
        let input_resolution = self.resolution();

        // angular wavenumber aka radians per meter
        let k = 2.0 * ::std::f64::consts::PI / self.wavelength;

        for ((yi, xi), ei) in self.amplitude.indexed_iter_mut() {
            let x = (xi as f64 - (wi - 1.0) / 2.0) * input_resolution;
            let y = (yi as f64 - (hi - 1.0) / 2.0) * input_resolution;
            let l2 = x * x + y * y + z * z;
            let l = l2.sqrt();

            *ei *= Complex::new(0.0, -k * l).exp();
        }
    }

    fn apply_circular_apeture(&mut self) {
        let max = (self.max_dim() as f64 - 1.0) / 2.0;
        // input height and width
        let hi = self.amplitude.shape()[0] as f64;
        let wi = self.amplitude.shape()[1] as f64;

        for ((y, x), e) in self.amplitude.indexed_iter_mut() {
            let x = x as f64 - (wi - 1.0) / 2.0;
            let y = y as f64 - (hi - 1.0) / 2.0;
            if y * y + x * x > max * max {
                *e *= Complex::new(0.0, 0.0);
            }
        }
    }

    fn max_dim(&self) -> usize {
        max(self.amplitude.shape()[0], self.amplitude.shape()[1])
    }

    fn resolution(&self) -> f64 {
        self.apeture_diameter / (self.max_dim() - 1) as f64
    }
}

/// reference size of the airy disk
/// distance from the peak to the first minima
///
/// 1.22 * λ * f / D
fn airy_radius(wavelength: f64, focal_length: f64, apeture_diameter: f64) -> f64 {
    1.22 * wavelength * focal_length / apeture_diameter
}

/// Conservative criteria to ensure high order components excluded in fresnel approximate
/// do not impact result. Must be much less than 1 to ensure accuracy.
/// dx and dy and the maximum possible differences in x and y space between input and output planes
fn fresnel_validity(lambda: f64, z: f64, dx: f64, dy: f64) -> f64 {
    let a = dx * dx + dy * dy;
    PI * a * a / (z * z * z * 4.0 * lambda)
}

pub (crate) struct OutputAmplitude {
    amplitude: Array2<Complex<f64>>,
    resolution: f64,
    z_position: f64,
}

// fn angular_spectrum_method(u, z){
//     u = ... //# this is your 2D complex field that you wish to propagate
//     z = ... //# this is the distance by which you wish to propagate

//     dx, dy = 1, 1 # or whatever

//     wavelen = 1 # or whatever
//     wavenum = 2 * np.pi / wavelen
//     wavenum_sq = wavenum * wavenum

//     kx = np.fft.fftfreq(u.shape[0], dx / (2 * np.pi))
//     ky = np.fft.fftfreq(u.shape[1], dy / (2 * np.pi))

//     # this is just for broadcasting, the "indexing" argument prevents NumPy
//     # from doing a transpose (default meshgrid behaviour)
//     kx, ky = np.meshgrid(kx, ky, indexing = 'ij', sparse = True)

//     kz_sq = kx * kx + ky * ky

//     # we zero out anything for which this is not true, see many textbooks on
//     # optics for an explanation
//     mask = wavenum * wavenum > kz_sq

//     g = np.zeros((len(kx), len(ky)), dtype = np.complex_)
//     g[mask] = np.exp(1j * np.sqrt(wavenum_sq - kz_sq[mask]) * z)

//     res = np.fft.ifft2(g * np.fft.fft2(u)) # this is the result
// }

// fn angular(){
//     clear all;
//     close all;
//     %% Example: Circular aperture.

//     lambda    = 632.8e-9; % 632.8nm wavelength light.
//     diameter  = 100.0e-6;   % 100um circular aperture diameter.
//     aperturez = 0;        % Aperture is located at z=0.
//     imagez    = 10.0e-2;   % 100cm from the aperture to the image plane.

//     aperture_sample_width = 5000.0e-6; % Sample the aperture for a width of 0.5mm.
//     % image_sample_width  = 2e-2; % Sample the image plane for a width of 2cm.

//     %nsamples = 1024; % Sample count (prefer power of 2).
//     nsamples =1024;

//     predict_airy_disc_radius = 1.22 * lambda * (imagez - aperturez) / diameter;
//     fprintf('Predicted Airy disc radius: %0.2f mm\n', ...
//             1e3 * predict_airy_disc_radius);

//     figure
//     colormap(gray)

//     % Plot the magnitude of the field at the (circular) aperture plane.
//     aperture_field = @(u,v) (sqrt((u.^2 + v.^2)) < (diameter/2));
//     delta_spacing = aperture_sample_width/nsamples
//     aperture_grid  = linspace(-aperture_sample_width/2-delta_spacing, ...
//                                aperture_sample_width/2, nsamples);
//     [ugrid, vgrid] = meshgrid(aperture_grid);
//     subplot(2,2,1)
//     imagesc(aperture_grid([1,end]) * 1e3, ...
//             aperture_grid([1,end]) * 1e3, ...
//             aperture_field(ugrid,vgrid))
//     axis square, xlabel 'mm', ylabel 'mm'

//     % Plot the magnitude of the angular spectrum at the aperture.
//     aperture_spectrum = ifft2(aperture_field(ugrid,vgrid));
//     aperture_spectrum_spacing = 1/aperture_sample_width;
//     aperture_spectrum_grid = ((1:nsamples) - (nsamples/2)-1) * ...
//                              aperture_spectrum_spacing;
//     [sx,sy] = meshgrid(aperture_spectrum_grid);
//     subplot(2,2,2)
//     imagesc(aperture_spectrum_grid([1,end]) / 1e3, ...
//             aperture_spectrum_grid([1,end]) / 1e3, ...
//             ifftshift(aperture_spectrum .* conj(aperture_spectrum)))
//     axis square, xlabel 'cycles / mm', ylabel 'cycles / mm'
//     %imagesc(ifftshift(aperture_spectrum .* conj(aperture_spectrum)))

//     % Plot the magnitude of the angular spectrum at the image plane.
//     % Shift this aperture spectrum to align with sx, sy grid.
//     image_spectrum_2 = fftshift(aperture_spectrum).* ...
//         exp((-1j * (2*pi/lambda) * (imagez - aperturez)) + ...
//            ( 1j * pi*lambda * (sx.^2 + sy.^2) * (imagez - aperturez)));

//     subplot(2,2,4)
//     %imagesc(aperture_spectrum_grid([1,end]) / 1e3, ...
//     %        aperture_spectrum_grid([1,end]) / 1e3, ...
//     %        ifftshift(image_spectrum .* conj(image_spectrum)))
//     imagesc(aperture_spectrum_grid([1,end]) / 1e3, ...
//             aperture_spectrum_grid([1,end]) / 1e3, ...
//             image_spectrum_2 .* conj(image_spectrum_2))

//     axis square, xlabel 'cycles / mm', ylabel 'cycles / mm'

//     % Plot the magnitude of the field at the image plane.
//     image_spectrum = fftshift(image_spectrum_2); % shift back to default
//     image_field = fft2(image_spectrum);
//     image_field_mag = image_field .* conj(image_field);
//     subplot(2,2,3)
//     imagesc(aperture_grid([1,end]) * 1e3, ...
//             aperture_grid([1,end]) * 1e3, ...
//             image_field_mag)
//     axis square, xlabel 'mm', ylabel 'mm'

//     % Plot the point spread function (PSF).
//     figure
//     plot(aperture_grid* 1e3, ...
//         (image_field_mag(nsamples/2,:) / max(image_field_mag(nsamples/2,:))))
//     axis square, xlabel 'mm', ylabel 'normalized intensity'
// }

fn rayleigh_sommerfeld(
    input: &InputAmplitude,
    width: usize,
    resolution: f64,
    z_position: f64,
) -> OutputAmplitude {
    let mut amplitude = Array2::from_elem([width, width], Complex::new(0.0, 0.0));
    let input_resolution = input.resolution();
    let z = z_position;

    // input height and width
    let hi = input.amplitude.shape()[0] as f64;
    let wi = input.amplitude.shape()[1] as f64;

    // output height and width
    let ho = amplitude.shape()[0] as f64;
    let wo = amplitude.shape()[1] as f64;

    // angular wavenumber aka radians per meter
    let k = 2.0 * ::std::f64::consts::PI / input.wavelength;

    let count = AtomicUsize::new(0);

    Zip::indexed(&mut amplitude).par_apply(|(yo, xo), eo| {
        let s = (xo as f64 - (wo - 1.0) / 2.0) * resolution;
        let t = (yo as f64 - (ho - 1.0) / 2.0) * resolution;

        for ((yi, xi), ei) in input.amplitude.indexed_iter() {
            let x = (xi as f64 - (wi - 1.0) / 2.0) * input_resolution;
            let y = (yi as f64 - (hi - 1.0) / 2.0) * input_resolution;
            let l2 = (x - s) * (x - s) + (y - t) * (y - t) + z * z;
            let l = l2.sqrt();
            let k_term = Complex::new(1.0 / (k * l), -1.0);

            *eo += *ei * k_term * Complex::new(0.0, k * l).exp() * Complex::new(z / l2, 0.0);
        }
        *eo /= input_resolution * input_resolution;

        let count = count.fetch_add(1, Ordering::AcqRel) + 1;
        println!("{:.3}%", count as f64 / (width * width) as f64 * 100.0);
    });

    OutputAmplitude {
        amplitude,
        resolution,
        z_position,
    }
}
