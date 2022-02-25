import numpy as np
import scipy.special as special
import datetime
import time
import h5py
import argparse
import pyiris.utils.utils as utils


def bake_swht_coeffs(array_name, center_freq, ant_coords, fov=np.array([[0, 360], [0, 90]]), resolution=1.0, lmax=85):
    """
    Makes an array containing all the factors that do not change with Visibility values.
    This array can then be saved to quickly create Brightness values given changing
    Visibilities. The array is then stored as a HDF5 file.

    Parameters
    ----------
        array_name : str
            Name of the antenna array. Used for file creation.
        center_freq : float
            Center frequency of the signal in [Hz].
        ant_coords : float np.array
            [[x0, x1, ...], [y0, y1, ...], [z0, z1, ...]] Antenna location in local Cartesian coordinates
            measured from antenna 0 in meters [m].
        fov : float np.array
            [[start, stop], [start, stop]] azimuth, elevation angles within 0 to 360 and 0 to 180 degrees.
        resolution : float
            Angular resolution in degree per pixel.
        lmax : int
            The maximum harmonic degree.

    Returns
    -------
        None

    Notes
    -----
        The array file must contain:
        wavelength : float
            Radar signal wavelength in meters.
        u : float np.array
            East-West baseline coordinate divided by wavelength.
        v : float np.array
            North-South baseline coordinate divided by wavelength.
        w : float np.array
            Altitude baseline coordinate divided by wavelength.
    """

    wavelength = 299792458.0 / center_freq
    date_created = datetime.datetime.now(datetime.timezone.utc)
    u, v, w = utils.baselines(ant_coords[0, :],
                              ant_coords[1, :],
                              ant_coords[2, :],
                              wavelength)
    ko = 2 * np.pi / wavelength
    az_step = int(np.abs(fov[0, 0] - fov[0, 1]) / resolution)
    el_step = int(np.abs(fov[1, 0] - fov[1, 1]) / resolution)
    r, t, p = utils.uvw_to_rtp(u, v, w)
    r *= wavelength  # Since r, t, p was converted from u, v, w we need the * wavelength back to match SWHT algorithm
    az = np.deg2rad(np.linspace(fov[0, 0], fov[0, 1], az_step))
    el = np.deg2rad(np.linspace(fov[1, 0], fov[1, 1], el_step))
    setting_name = f"{int(np.round(np.abs(fov[0, 0] - fov[0, 1]))):03d}az_" \
                   f"{int(np.round(np.abs(fov[1, 0] - fov[1, 1]))):03d}el_" \
                   f"{str(resolution).replace('.', '')}res_" \
                   f"{lmax}lmax"
    filename = f"swhtcoeffs_{array_name}_{date_created.year}_{date_created.month}_{date_created.day}" \
               f"_{date_created.hour}_{setting_name}.h5"

    print(f"Calculating SWHT coeffs:")
    print(f"\t-filename: {filename}")
    print(f"\t-date: {date_created.year}:{date_created.month}:{date_created.day}:{date_created.hour}")
    print(f"\t-configuration: {array_name}")
    print(f"\t-azimuth: {fov[0, 0]} - {fov[0, 1]}")
    print(f"\t-elevation: {fov[1, 0]} - {fov[1, 1]}")
    print(f"\t-resolution: {resolution}")
    print(f"\t-degree: {lmax}")
    print(f"\t-wavelength: {wavelength}")

    date_created = np.array([date_created.year, date_created.month, date_created.day, date_created.hour])
    create_coeffs(filename, date_created, array_name, fov, resolution, lmax, wavelength, np.array([u, v, w]))
    calculate_coeffs(filename, az, el, ko, r, t, p, lmax)

    return filename


def create_coeffs(filename, date_created, array_name, fov, resolution, lmax, wavelength, baselines):
    f = h5py.File(filename, 'w')
    f.create_dataset('radar_config', data=np.array(array_name, dtype='S'))
    f.create_dataset('date_created', data=date_created)
    f.create_dataset('fov', data=fov)
    f.create_dataset('resolution', data=resolution)
    f.create_dataset('lmax', data=lmax)
    f.create_dataset('wavelength', data=wavelength)
    f.create_dataset('baselines', data=baselines)
    f.create_group('coeffs')
    f.close()

    return None


def append_coeffs(filename, l, coeffs):
    f = h5py.File(filename, 'a')
    f.create_dataset(f'coeffs/{l:02d}', data=coeffs)
    f.close()

    return None


def calculate_coeffs(filename, az, el, ko, r, t, p, lmax=85):
    """
    Makes an array containing all the factors that do not change with Visibility values.
    This array can then be saved to quickly create Brightness values given changing
    Visibilities. The array is then stored as a HDF5 file.

    Parameters
    ----------
        filename : string
            Filename and path to the HDF5 file the calculated coefficients are to be appended.
        az : float np.array
            An array of azimuth angles in radians to calculate coefficients for.
        el : float np.array
            An array of elevation angles in radians to calculate coefficients for.
        lmax : int
            The maximum harmonic degree.
        ko : float
            Radar signal wave number, ko = 2pi/wavelength.
        r : float np.array
            Radius baseline coordinate divided by wavelength.
        t : float np.array
            Theta (elevation) baseline coordinate.
        p : float np.array
            Phi (azimuthal) baseline coordinate.

    Returns
    -------
        None

    Notes
    -----
        Maximum harmonic degree is Lmax = 85. Above this scipy crashes due to an overflow error. The potential fix is to
        scale the initial Pmm of the recursion by 10^280 sin^m (theta), and then rescale everything back at the end.

        Holmes, S. A., and W. E. Featherstone, A unified approach to the Clenshaw summation and the recursive
        computation of very high degree and order normalised associated Legendre functions,
        J. Geodesy, 76, 279- 299, doi:10.1007/s00190-002-0216-2, 2002.
    """

    start_time = time.time()
    AZ, EL = np.meshgrid(az, el)
    coeffs = np.zeros((len(el), len(az), len(r)), dtype=np.complex128)

    if lmax <= 85:
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                coeffs += ko ** 2 / (2 * np.pi ** 2 * np.round((-1j) ** l)) * \
                          np.repeat(special.sph_harm(m, l, AZ, EL)[:, :, np.newaxis], len(r), axis=2) * \
                          np.repeat(np.repeat(special.spherical_jn(l, ko * r) * \
                                              np.conjugate(special.sph_harm(m, l, p, t)) \
                                                  [np.newaxis, np.newaxis, :], AZ.shape[0], axis=0), AZ.shape[1],
                                    axis=1)
                print(f"\tharmonic degree (l) = {l:02d}/{lmax:02d}, order (m) = {m:02d}/{l:02d}\r")
            if l in [5, 15, 25, 35, 45, 55, 65, 75, 85]:
                append_coeffs(filename, l, coeffs)

    # This section has not been proven to work. lmax > 85 should not be used.
    elif lmax > 85:
        try:
            import pyshtools as pysh
        except ImportError:
            print(f'Error: lmax = {lmax} -- values over 85 requires PySHTOOLS '
                  f'https://github.com/SHTOOLS try pip install pyshtools')
            exit()
        print(
            f'\twarning: lmax values over 85 generate massive files only 1/10th frames will be stored, evenly distributed')
        ylm_pysh = np.vectorize(pysh.expand.spharm_lm)
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                coeffs += ko ** 2 / (2 * np.pi ** 2 * np.round((-1j) ** l)) * \
                          np.repeat(
                              ylm_pysh(l, m, EL, AZ, normalization='ortho', csphase=-1, kind='complex', degrees=False)[
                              :, :, np.newaxis], len(r), axis=2) * \
                          np.repeat(np.repeat(special.spherical_jn(l, ko * r) * \
                                              np.conjugate(ylm_pysh(l, m, t, p, normalization='ortho', csphase=-1,
                                                                    kind='complex', degrees=False)) \
                                                  [np.newaxis, np.newaxis, :], AZ.shape[0], axis=0), AZ.shape[1],
                                    axis=1)
                print(f"\tharmonic degree (l) = {l:02d}/{lmax:02d}, order (m) = {m:02d}/{l:02d}\r")
            if l == 85:
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.1):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.2):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.3):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.4):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.5):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.6):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.7):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.8):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.9):
                append_coeffs(filename, l, coeffs)
        append_coeffs(filename, l, coeffs)

    print(f"Complete time: \t{time.time() - start_time} s")

    return None


def unpackage_coeffs(filename, index):
    """
    Unpack the coefficient values from a given hdf5 file of pre-calculated SWHT coefficients.

    Parameters
    ----------
        filename : str
            File and path to the HDF5 file containing the SWHT coefficients to unpack.
        index :
            The harmonic order index to unpack.

    Returns
    -------
        coeffs : complex128 np.array
            Complex matrix of coefficients for the SWHT with dimension fov / resolution.
    """

    f = h5py.File(filename, 'r')
    try:
        coeffs = np.array(f['coeffs'][f'{index:02d}'][()], dtype=np.complex64)
    except:
        coeffs = np.array(f['coeffs'][()], dtype=np.complex64)
    print(f'hdf5 coeffs: index = {index}, shape = {coeffs.shape}')
    return coeffs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arrayname', type=str, help="Name of the array; used for file naming.")
    parser.add_argument('-f', '--frequency', type=float, help="Frequency to generate coefficient matrix at.")
    parser.add_argument('-c', '--coordinates', type=list, help="Cartesian basis local antenna coordinates in meters.")
    parser.add_argument('-v', '--fov', type=list, help="Field of view to generate coefficients over.")
    parser.add_argument('-r', '--resolution', type=float, help="Resolution (degree per pixel) to create images.")
    parser.add_argument('-l', '--lmax', type=int, help="Maximum spherical harmonic order.")
    args = parser.parse_args()
    # TODO: Fix this. Antenna coords needs to me an nx3 numpy array but thats not working from cli.
    # bake_swht_coeffs(args.arrayname, args.frequency, np.asarray(args.coordinates), args.fov, args.resolution, args.lmax)
    bake_swht_coeffs("test", 50e6, np.array([[1, 2, 3],[2, 4, 5],[1, 2, 5]]), lmax=5)