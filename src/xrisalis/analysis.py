import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as ft
import scipy.constants as spc
from astropy.io import fits
from matplotlib.colors import LinearSegmentedColormap


def hist_data(data, binsno, pixs=False, num=0):
    """
    Function that makes a histogram of direct output data from an interferometer object.

    Parameter:
    data (interferometer_data class object): Data to be plotted.
    binsno (int): Number of bins in the histogram.
    pixs (Boolean): whether the x-axis is in units of pixels or meters. If true, then in pixels. Default is False.
    """

    if pixs:
        plt.hist(data, binsno, label=f"Baseline {num}")
        plt.xlabel("Detector position (pixels)")
    else:
        plt.hist(data, binsno, label=f"Baseline {num}")
        plt.xlabel("Detector position (micrometers)")
    plt.ylabel("Counts")


def ft_data(y_data, samples, spacing):
    """
    Function that fourier transforms given input data from an interferometer.
    Works by first making a histogram of the positional data to then fourier transform that and obtain spatial frequencies.

    Parameters:
    data (interferometer_data class object): Data to be fourier transformed.
    samples (int): Number of samples for the fourier transform to take.
    """
    ft_x_data = ft.fftfreq(samples, spacing)
    ft_y_data = ft.fft(y_data) / y_data.size

    return ft_x_data, ft_y_data


def plot_ft(ft_x_data, ft_y_data, plot_obj, log=0, num=0):
    """
    Function to plot fourier transformed interferometer data in a number of ways.

    Parameters:
    ft_x_data (array): fourier transformed data for the x-axis (so spatial frequencies).
    ft_y_data (array): fourier transformed data for the y-axis.
    log (int in [0,2]): indicates how many axes are to be in log scale, with 1 having only the y-axis in log.
    """
    if log == 0:
        plot_obj.plot(
            ft.fftshift(ft_x_data), (ft.fftshift(ft_y_data)), label=f"Baseline {num}"
        )
    if log == 1:
        plot_obj.semilogy(
            ft.fftshift(ft_x_data), (ft.fftshift(ft_y_data)), label=f"Baseline {num}"
        )
    if log == 2:
        plot_obj.loglog(
            ft.fftshift(ft_x_data), (ft.fftshift(ft_y_data)), label=f"Baseline {num}"
        )


def image_recon_smooth(
    data,
    instrument,
    fov,
    samples=np.array([512, 512]),
    progress=0,
    error=0.1,
    recon_type="IFFT",
    verbose=True,
):
    """
    This function is to be used to reconstruct images from interferometer data.
    Bins input data based on roll angle, which is important to fill out the uv-plane that will be fed into
    the 2d inverse fourier transform.

    Args:
        data (interferometer_data class object): The interferometer data to recover an image from.
        instrument (interferometer class object): The interferometer used to record the aforementioned data.
        samples (int): N for the NxN matrix that is the uv-plane used for the 2d inverse fourier transform. TODO: outdated
        progress (1 or 0): whether, or not respectively, to show progess of image reconstruction calculations.
        TODO: add fov argument, unknown at this time; largest axis size in arcsec

    Returns:
        array, array, array: Three arrays, first of which is the reconstructed image,
                                second and third of which are the fourier transforms and associated uv coordinates of each roll bin + energy channel combination.
    """

    def inverse_fourier(
        f_values, uv, fov, error, instrument, energies, recon_type, verbose
    ):
        """
        This is a helper function that calculates the inverse fourier transform of the data from all baselines, only to
        be used at the last step of the parent function. It is sectioned off here for legibility.

        It first defines an image to fill in according to the provided samples size, and calculates the sum
        of all fourier components over the whole image.
        """

        if recon_type == "IFFT":
            wavelengths = spc.h * spc.c / energies

            d = (
                (fov * np.pi / (180 * 3600)) / np.max(samples)
            )  # 1 / (2 * (np.max(np.array([baseline.D for baseline in instrument.baselines])) / np.min(wavelengths)))
            n = np.ceil(
                1
                / (
                    2
                    * d
                    * (
                        np.min(
                            np.array([baseline.D for baseline in instrument.baselines])
                        )
                        / np.max(wavelengths)
                    )
                    * error
                )
            ).astype(int)  # + 1

            if verbose:
                print(
                    f"Calculated image parameters:\n sample spacing: {d} \n window size: {n} "
                )

            # print(d)
            # print(1 / (2 * (np.max(np.array([baseline.D for baseline in instrument.baselines])) / np.min(wavelengths))))

            # calculate the frequency grid onto which the real sampled frequencies are mapped
            fft_freqs = np.fft.fftfreq(n, d)
            if verbose:
                print("Fourier frequencies calculated.")

            # find closest frequency grid point in numpy.fft.fftfreq of the real sampled frequencies,
            # dividing by frequency bin width: Delta f = 1 / (n * d)
            u_conv = np.around(uv[:, 0] * (n * d)).astype(int)
            v_conv = np.around(uv[:, 1] * (n * d)).astype(int)
            uv_conv = np.array([u_conv, v_conv]).T
            # create nxn matrix, fourier space image
            image = np.zeros(
                (fft_freqs.size, fft_freqs.size), dtype=np.complex64
            )  # complex)#np.zeros((fft_freqs_u.size, fft_freqs_v.size), dtype = complex)#np.zeros(samples, dtype = complex)

            # add all fourier values to the corresponding frequency coordinate as per numpy.fft.fftfreq
            np.add.at(image, (u_conv, v_conv), f_values)
            if verbose:
                print("Fourier values added to image matrix. Performing inverse FFT...")

            fft_image = np.fft.ifft2(image)

            fft_image = np.real(fft_image)
            if verbose:
                print("Inverse FFT complete. Shifting image...")
            fft_image = np.roll(
                fft_image, int(np.around(fft_image.shape[0] / 2)), axis=0
            )
            fft_image = np.roll(
                fft_image, int(np.around(fft_image.shape[1] / 2)), axis=1
            )

            x_npix_real_image = samples[0]
            x_lower_bound = np.round(
                (fft_image.shape[0] / 2) - (0.5 * x_npix_real_image)
            ).astype(int)
            x_upper_bound = np.round(
                (fft_image.shape[0] / 2) + (0.5 * x_npix_real_image)
            ).astype(int)
            y_npix_real_image = samples[1]
            y_lower_bound = np.round(
                (fft_image.shape[0] / 2) - (0.5 * y_npix_real_image)
            ).astype(int)
            y_upper_bound = np.round(
                (fft_image.shape[0] / 2) + (0.5 * y_npix_real_image)
            ).astype(int)

            # prevent indexing problems
            if x_lower_bound < 0:
                x_lower_bound = 0
            if x_upper_bound >= fft_image.shape[0]:
                x_upper_bound = fft_image.shape[0] - 1
            if y_lower_bound < 0:
                y_lower_bound = 0
            if y_upper_bound >= fft_image.shape[0]:
                y_upper_bound = fft_image.shape[0] - 1

            if verbose:
                print("Image shifted. Returning image...")

            return (
                (
                    fft_image[
                        np.ix_(
                            np.arange(x_lower_bound, x_upper_bound, 1),
                            np.arange(y_lower_bound, y_upper_bound, 1),
                        )
                    ],
                    (samples.max() * (1e6 * 3600 * 360 / (2 * np.pi)) * (d * n)),
                ),
                image,
                uv_conv,
                fft_freqs,
            )

        # else:

        #     def inverse_fourier_val(x, y, v, u, fourier):

        #         # This function is the formula for an inverse fourier transform, without the integration.
        #         # It is included here to make clear that a discrete inverse fourier transform is what is happening, and
        #         # to make clear what argument means what and for multi-threading.

        #         global re_im

        #         # frequency shift + 1 / (d * n)
        #         re_im += fourier * np.exp(2j * np.pi * ((u) * x + (v) * y))

        #     global re_im
        #     shape = np.array(samples, dtype = int)
        #     re_im = np.zeros(shape, dtype = complex)

        #     fov_x = fov
        #     fov_y = fov_x*(shape[0]/shape[1]) # Assumes square pixels

        #     x_pos = np.linspace(-fov_x/2,fov_x/2,shape[1]+1)*2*np.pi/(360*3600)
        #     y_pos = np.linspace(-fov_y/2,fov_y/2,shape[0]+1)*2*np.pi/(360*3600)

        #     # Pixel positions for image calculation correspond to pixel centres:
        #     x_pix = (x_pos[:-1]+x_pos[1:])/2
        #     y_pix = (y_pos[:-1]+y_pos[1:])/2
        #     x_grid, y_grid = np.meshgrid(x_pix, y_pix, indexing='xy')

        #     usefull = np.nonzero(f_values)[0].astype(int)

        #     # multi threading
        #     thread_list = []
        #     for i, ft_val in enumerate(f_values[usefull]):
        #         thread = threading.Thread(target = inverse_fourier_val, args = (x_grid, y_grid, uv[usefull][i,0], uv[usefull][i,1], ft_val))
        #         thread.start()
        #         thread_list.append(thread)

        #     # waiting for all the threads to finish
        #     for thread in thread_list:

        #         thread.join()

        #     return np.real(re_im), f_values, uv, uv

    # These arrays are all copied locally to reduce the amount of cross-referencing to other objects required.
    time_data = data.toa
    E_data = data.energies
    base_ind = data.baseline_indices
    pointing = data.pointing
    positional_data = data.pos

    # Generating the arrays that will contain the uv coordinates and associated fourier values covered by the interferometer.
    uv = []
    f_values = np.array([])

    # Looking only at the baselines that have associated photons
    for k in np.unique(base_ind):
        if verbose:
            print(f"Processing baseline {k + 1} / {len(instrument.baselines)}")

        # Taking only relevant photons from the current baseline
        in_baseline = base_ind == k

        baseline = instrument.baselines[k]

        # Calculating the wavelength of light we are dealing with, and the frequency that this baseline covers in the uv-plane with it.
        lam_baseline = spc.h * spc.c / E_data[in_baseline]
        # ensure only unique energies are used, to avoid calculating all photons seperately when possible
        freq_baseline = baseline.D / lam_baseline

        # Calculating the frequency we will be doing the fourier transform for, which is the frequency we expect the fringes to appear at.
        fourier_freq = 1 / (lam_baseline / baseline.beam_angle)

        # the photon positions in the bin
        data_bin_roll = positional_data[in_baseline]

        if verbose:
            print(
                f"Number of photons in baseline: {data_bin_roll.size}\nCalculating uv coordinates and fourier values..."
            )

        # Calculating u and v for middle of current bin by taking a projection of the current frequency
        u = (freq_baseline + 0) * np.sin(
            pointing[time_data, 2][in_baseline] % (2 * np.pi)
        )
        v = (freq_baseline + 0) * np.cos(
            pointing[time_data, 2][in_baseline] % (2 * np.pi)
        )
        new_uv_pairs = np.array([u, v]).T

        # Doing the same with the negative frequency
        uv.extend(
            np.column_stack((new_uv_pairs, -new_uv_pairs)).reshape(
                -1, new_uv_pairs.shape[1]
            )
        )

        # Calculating value of the fourier transform for the current frequency and bin
        f_value = np.exp(-2j * np.pi * fourier_freq * data_bin_roll)

        # Doing the same with the negative frequency
        f_values = np.append(f_values, np.ravel([f_value, np.conjugate(f_value)], "F"))

        if verbose:
            print(f"Baseline {k + 1} processed.\n")

    # reshaping the uv coordinate matrix
    uv = np.array(uv).reshape(-1, 2)

    if verbose:
        print("All baselines processed. Starting IFFT...")

    # create image from IFT
    img, ft_img, uv_conv, fft_freqs = inverse_fourier(
        f_values, uv, fov, error, instrument, E_data, recon_type, verbose=verbose
    )

    if verbose:
        print("Image reconstruction complete.")
    # store results in a reconstruction_data object
    data_obj = reconstruction_data(
        img, ft_img, f_values, uv, uv_conv, fft_freqs if recon_type == "IFFT" else None
    )

    return img, uv, data_obj


class reconstruction_data:
    """
    This class is to be used to store the data that is used for the image reconstruction.
    It contains the image, the fourier transform of the image, and the uv coordinates of the fourier transform.
    """

    def __init__(self, image, ft_image, f_values, uv, uv_conv, fft_freqs=None):
        self.image = image
        self.ft_image = ft_image
        self.f_values = f_values
        self.uv = uv
        self.uv_conv = uv_conv
        self.fft_freqs = fft_freqs


def plot_spec_per_baseline(event_list, exposure_time):
    hdul = fits.open(event_list)
    event_data = hdul[1].data
    num_baselines = np.max(event_data["BASELINE_ID"]) + 1
    hdul.close()
    plt.figure(figsize=(7, 5))

    min_e = np.min(event_data["DETECTED_ENERGY"])
    max_e = np.max(event_data["DETECTED_ENERGY"])
    bins = np.linspace(min_e, max_e, 101)  # 100 bins
    bin_width = bins[1] - bins[0]

    weights = np.ones_like(event_data["DETECTED_ENERGY"]) / (exposure_time * bin_width)

    plt.hist(
        event_data["DETECTED_ENERGY"],
        bins=bins,
        weights=weights,
        histtype="step",
        log=True,
        label="All Baselines",
        color="darkblue",
        lw=2,
    )

    for baseline_id in range(num_baselines):
        mask = event_data["BASELINE_ID"] == baseline_id
        subset_weights = np.ones_like(event_data["DETECTED_ENERGY"][mask]) / (
            exposure_time * bin_width
        )
        plt.hist(
            event_data["DETECTED_ENERGY"][mask],
            bins=bins,
            weights=subset_weights,
            histtype="step",
            label=f"Baseline {baseline_id + 1}",
            color="darkblue",
            alpha=1 / (baseline_id + 1),
            lw=1,
        )
    plt.xlim(min_e, max_e)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Energy (keV)")
    plt.ylabel("cts/s/keV")
    plt.legend(fontsize=12, loc="best")
    plt.grid("both")
    plt.show()


colors_chandra = [
    "#000000",  # very dark
    "#242037",  # deep navy
    "#344871",  # dark blue
    "#89A4E3",  # mid-to-light blue
    "#E2F6FE",  # bright white-blue
]

xri_cmap = LinearSegmentedColormap.from_list("black_to_blue", colors_chandra, N=1e4)


def plot_inf_image(image, fov_arcsec, return_fig=False):
    fig = plt.figure()
    image = image[0]
    fov_uas = fov_arcsec * 1e6
    extent = [-fov_uas / 2, fov_uas / 2, -fov_uas / 2, fov_uas / 2]
    plt.imshow(
        image / np.max(np.abs((image))),
        extent=extent,
        cmap=xri_cmap,
        origin="lower",
        vmin=0,
        vmax=1,
    )
    plt.xlabel("$x$ ($\\mu$as)")
    plt.ylabel("$y$ ($\\mu$as)")
    plt.colorbar(label="Intensity (normalized)")
    plt.title("Reconstructed image")
    plt.show()
    if return_fig:
        return fig


def plot_uv(uv, return_fig=False):
    uv_plot = uv / 1e9  # convert to Giga-lambda

    # plot uv-plane sampling
    fig = plt.figure(figsize=(6, 6))
    plt.plot(uv_plot[:, 0], uv_plot[:, 1], "ro", markersize=1)
    # plt.xlim(-np.max(uv_plot) * 1.2, np.max(uv_plot) * 1.2)
    # plt.ylim(-np.max(uv_plot) * 1.2, np.max(uv_plot) * 1.2)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.title("$uv$-plane sampling")
    plt.xlabel("$u$ (G$\\lambda$)")
    plt.ylabel("$v$ (G$\\lambda$)")
    plt.show()
    if return_fig:
        return fig


def plot_res_hist(uv, return_fig=False):
    uv_abs = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
    interf_res = 1 / (2 * uv_abs)  # in microarcseconds
    interf_res_uas = interf_res * 1e6 * 3600 * 180 / np.pi  # convert to microarcseconds
    bins = np.logspace(
        np.log10(np.min(interf_res_uas)), np.log10(np.max(interf_res_uas)), 200
    )
    fig = plt.figure(figsize=(7, 5))
    plt.hist(
        interf_res_uas, bins=bins, color="darkred", alpha=0.7, histtype="stepfilled"
    )
    plt.xlabel("Interferometric resolution ($\\mu$as)")
    plt.ylabel("$\\#$Photons")
    plt.title("Distribution of interferometric resolution")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.show()
    if return_fig:
        return fig


def subplots(image, fov_arcsec, uv, return_fig=False):
    fig, axs = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]}
    )

    # Plot the reconstructed image
    img_array = image[0]
    fov_uas = fov_arcsec * 1e6
    extent = [-fov_uas / 2, fov_uas / 2, -fov_uas / 2, fov_uas / 2]
    im = axs[0].imshow(
        img_array / np.max(np.abs(img_array)),
        extent=extent,
        cmap=xri_cmap,
        origin="lower",
        vmin=0,
        vmax=1,
    )
    axs[0].set_xlabel("$x$ ($\\mu$as)")
    axs[0].set_ylabel("$y$ ($\\mu$as)")
    axs[0].set_title("Reconstructed image")
    fig.colorbar(
        im, ax=axs[0], label="Intensity (normalized)", fraction=0.046, pad=0.04
    )

    # Plot the uv-plane sampling
    uv_plot = uv / 1e9  # convert to Giga-lambda
    axs[1].plot(uv_plot[:, 0], uv_plot[:, 1], "ro", markersize=1, alpha=0.5)
    axs[1].set_xlim(-6, 6)
    axs[1].set_ylim(-6, 6)
    axs[1].set_aspect("equal")
    axs[1].set_title("$uv$-plane sampling")
    axs[1].set_xlabel("$u$ (G$\\lambda$)")
    axs[1].set_ylabel("$v$ (G$\\lambda$)")

    plt.tight_layout()
    plt.show()

    if return_fig:
        return fig


def filter_data(data, E_bounds, baseline_bounds=None):
    if baseline_bounds is not None:
        baseline_mask = (data.baseline_indices >= baseline_bounds[0]) & (
            data.baseline_indices <= baseline_bounds[1]
        )
    else:
        baseline_mask = np.ones_like(data.baseline_indices, dtype=bool)
    mask = (
        (data.energies / 1e3 / spc.e > E_bounds[0])
        & (data.energies / 1e3 / spc.e < E_bounds[1])
    ) & baseline_mask
    data_temp = data.copy()
    data_temp.pos = data.pos[mask]
    data_temp.energies = data.energies[mask]
    data_temp.baseline_indices = data.baseline_indices[mask]
    data_temp.size = len(data_temp.pos)
    data_temp.toa = data.toa[mask]
    print(f"Events in filtered data: {len(data_temp.pos)}")
    return data_temp
