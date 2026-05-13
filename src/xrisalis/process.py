import warnings
import numpy as np
import scipy.special as sps
import scipy.stats as sampler
import scipy.constants as spc
import scipy.interpolate as interp
from astropy.io import fits

# for testing
import matplotlib.pyplot as plt


class interferometer_data:
    """
    Class that serves as a container for interferometer output data.
    Constructed as such for ease of use by way of standardization.
    Does not contain manipulation methods, data inside will have to be edited via external methods.
    """

    def __init__(
        self,
        instrument,
        image,
        eff_area_show=True,
        pure_diffraction=False,
        pure_fringes=False,
    ):
        """
        This function is the main function that takes an image and converts it to instrument data as
        if the instrument had just observed the object the image is a representation of.
        It models the individual photons coming in each timestep, at what detector they end up,
        whether they are absorbed along the way, how much noise there is, and also whether the
        spacecraft the instrument is on wobbles, and possible correction for this.

        Parameters:\n

        instrument (interferometer class object) = Instrument object to be used to simulate observing the image.\n
        image (image class object) = Image object to be observed.\n

        eff_area_show (bool) = whether or not to show the effective area of the full instrument, mirrors and detector\n
        pure_diffraction (bool) = sample only the diffraction pattern, without the fringe pattern\n
        pure_fringes (bool) = sample only the fringe pattern, without the diffraction pattern\n
        """

        # Useful shorthands
        self.size = image.size

        # image values, for later reference
        self.image_energies = image.energies
        self.image_toa = image.toa

        # simulates the energies of the source photons going through the optical benches until they reach the detector
        # self.actual_energies, self.baseline_indices = self.process_photon_e_base(instrument, image, eff_area_show)
        self.actual_energies, self.baseline_indices = (
            image.energies,
            image.baseline_indices,
        )
        # detector values, which can be subjected to detector effects
        self.energies = self.actual_energies
        self.toa = self.image_toa

        # simulates the positions (in m) of the source photons going through the optical benches until they reach the detector
        self.actual_pos = self.process_photon_dpos(
            instrument, image, pure_diffraction, pure_fringes
        )

        # detector value, which can be subjected to detector effects, digitize to pixel positions
        self.pos = np.round(self.actual_pos / 5e-6) * 5e-6

    def process_photon_e_base(self, instrument, image, eff_area_show=True):
        """
        This function is a helper function for process_image that specifically processes what energy detected photons, which impact
        on the detector, are expected. Not to be used outside the process_image context.

        Parameters:\n

        instrument (interferometer class object) = Instrument object to be used to simulate observing the image.\n
        image (image class object) = Image object to be observed.\n

        eff_area_show (bool) = whether or not to show the effective area of the full instrument, mirrors and detector\n
        """

        energies = self.image_energies

        array_lengths = self.size

        # Select random baseline based on effective area per basline
        if image.spectrum is None:
            # pure effective area calculation
            eff_areas = np.array(
                [
                    baseline.eff_area(np.unique(energies))
                    for baseline in instrument.baselines
                ]
            )
            relative_eff_areas = eff_areas / np.sum(eff_areas, axis=0)

            baseline_indices = np.random.choice(
                len(instrument.baselines), array_lengths, p=relative_eff_areas.T[0]
            )  # np.random.randint(0, len(instrument.baselines), array_lengths) #
        else:
            # effective area calculation, including source spectra

            cts_perbaseline_persec = []
            all_eCDF = []
            min_val_eCDF = []

            # loop through the baselines
            for baseline in instrument.baselines:
                # pull out detector for its attributes
                detector = baseline.camera

                # define the integration energies, 100 is used to integrate finer then the energy resolution
                # how much better energy sampling, compared to detector energy resolution
                finer_binning = 1e2
                integr_energy = np.linspace(
                    detector.E_range[0],
                    detector.E_range[1],
                    (
                        finer_binning
                        * np.ceil(
                            (detector.E_range[1] - detector.E_range[0]) / detector.res_E
                        )
                    ).astype(int),
                )

                # get the effective area of the baseline, per energy
                eff_area = baseline.eff_area(integr_energy).T

                # spectra assumed to use keV, turned into SI units of J
                interp_spec = interp.interp1d(
                    image.spectrum[:, 0] * 1e3 * spc.eV,
                    image.spectrum[:, 1],
                    bounds_error=False,
                    fill_value=0,
                )(integr_energy)

                # integrate the spectrum multiplied with the effective areas over energies, for photons per baseline
                cts_persec = np.sum(
                    interp_spec
                    * eff_area
                    * (integr_energy[-1] - integr_energy[0])
                    / integr_energy.size
                )
                cts_perbaseline_persec.append(cts_persec)

                # otherwise difficult to find error
                if (interp_spec == 0).all():
                    raise ValueError("The spectrum hasn't been sampled correctly!!")

                # inversion sampling of photon energies, eCDF calculations
                e_cumsum = np.cumsum(interp_spec * eff_area)
                e_eCDF = e_cumsum / e_cumsum[-1]
                min_val_eCDF.append(np.min(e_eCDF))
                all_eCDF.append(
                    interp.interp1d(e_eCDF, integr_energy)
                )  # , bounds_error = False, fill_value = np.nan))

            if eff_area_show:
                plt.plot(
                    integr_energy / (1e3 * spc.eV),
                    eff_area,
                    color="#ff7f0e",
                    label="Effective area",
                )
                plt.hlines(
                    np.power(detector.pos_range[1] - detector.pos_range[0], 2),
                    0,
                    10,
                    label="Real area",
                )
                # plt.title("Effective area curve of mirrors and detector")
                plt.ylabel("Effective area ($m^2$)")
                plt.xlabel("Energy ($keV$)")
                plt.xlim(1, 7)
                plt.ylim(1e-12, 1e-2)
                plt.yscale("log")
                # # plt.xscale("log")
                plt.legend()
                # plt.savefig("A_eff.pdf")
                plt.show()

            # Select random baseline based on effective area per basline
            baseline_indices = np.random.choice(
                len(instrument.baselines),
                array_lengths,
                p=cts_perbaseline_persec / np.sum(cts_perbaseline_persec, axis=0),
            )  # np.random.randint(0, len(instrument.baselines), array_lengths) #

            # Looking only at the baselines that have associated photons
            for index, baseline in enumerate(instrument.baselines):
                # Taking only relevant photons from the current baseline
                in_baseline = (
                    np.array(instrument.baselines)[baseline_indices] == baseline
                )

                # inversion sampling of photon energies, based on the eCDFs
                energies[in_baseline] = all_eCDF[index](
                    np.random.uniform(min_val_eCDF[index], 1, np.sum(in_baseline))
                )

        return energies, baseline_indices

    def fre_dif(self, wavelength, baseline, samples, y_pos=None):
        """
        Helper function that calculates the Fresnell difraction pattern for a beam
        such as the case in the interferometer. Also aplicable for, but less efficient at,
        Fraunhoffer diffraction patterns.

        Parameters:\n

        wavelength (array of floats) = the wavelenths for the photons.\n
        baseline (baseline object) = the baseline the photon(s) is/are in.\n
        samples (int) = the number of samples to take.\n
        y_pos (array of floats) = positions to calculate the fresnel diffraction value at.
        """

        # see Willingale (2004) for the definitions of the dimensionless coordinate u,
        # used for the Fresnel integrals
        u_0 = baseline.W * np.sqrt(2 / (wavelength * baseline.L))

        # Only sample the slit size, or set values
        if y_pos is None:
            y_pos = np.linspace(-baseline.W / 2, baseline.W / 2, int(samples))
        u = y_pos * np.sqrt(2 / (wavelength * baseline.L))

        u_1 = u - u_0 / 2
        u_2 = u + u_0 / 2

        # Fresnel integrals
        S_1, C_1 = sps.fresnel(u_1)
        S_2, C_2 = sps.fresnel(u_2)

        # Fresnel amplitude
        A = (C_2 - C_1) + 1j * (S_2 - S_1)

        # Fresnel intensity
        A_star = np.conjugate(A)
        Int = np.abs(A * A_star)

        return Int, u

    def process_photon_dpos(
        self, instrument, image, pure_diffraction=False, pure_fringes=False
    ):
        """
        This function is a helper function for process_image that specifically processes the locations where photons impact
        on the detector (hence the d(etector)pos(ition) name). Not to be used outside the process_image context.

        Parameters:\n

        instrument (interferometer class object) = Instrument object to be used to simulate observing the image.\n
        image (image class object) = Image object to be observed.\n

        pure_diffraction (bool) = sample only the diffraction pattern, without the fringe pattern\n
        pure_fringes (bool) = sample only the fringe pattern, without the diffraction pattern\n

        """

        array_lengths = self.size

        actual_pos = np.zeros(array_lengths)

        # Defining the pointing, relative position and off-axis angle for each photon over time.
        # Relative position is useful for the calculation of theta, since the off-axis angle is dependent on pointing position and orientation
        self.pointing = instrument.gen_pointing(int(np.max(self.image_toa)))
        pos_rel = self.pointing[self.image_toa, :2] - image.loc
        theta = np.cos(
            self.pointing[self.image_toa, 2] - np.arctan2(pos_rel[:, 0], pos_rel[:, 1])
        ) * np.sqrt(pos_rel[:, 0] ** 2 + pos_rel[:, 1] ** 2)

        # only populated baselines are selected
        for baseline_i in np.unique(self.baseline_indices):
            # which indices apply
            photons_in_baseline = self.baseline_indices == baseline_i

            # select baseline
            baseline = instrument.baselines[baseline_i]

            # allow for keeping track of rejected photon properties
            thetas = theta[photons_in_baseline]
            energies = self.actual_energies[photons_in_baseline]
            indices = np.arange(len(thetas))
            times = self.image_toa[photons_in_baseline]

            # initialize the sampled y positions
            y_pos = np.zeros(len(indices))

            # calculate all wavelengths
            wavelengths = spc.h * spc.c / energies

            # accept reject until all photons have been sampled
            while len(indices) > 0:
                # how many photons still need to be sampled in order to reach the specified number,
                # no photons are lost
                samples_left = len(indices)

                # range on the detector where photons can arrive
                width_box = np.array(
                    [
                        (-baseline.W * baseline.num_pairs / 2) - baseline.L * thetas,
                        (baseline.W * baseline.num_pairs / 2) - baseline.L * thetas,
                    ]
                )

                # Tested to be just above the maximum of the Fresnel diffraction,
                # for the height of the box needed by the accept reject method.
                height_box = [
                    0,
                    2.7 * self.fre_dif(wavelengths, baseline, 1, y_pos=0)[0],
                ]

                # the fringe spacing, see Willingale (2004) Eq. 1
                fringe_spacing = wavelengths / baseline.beam_angle

                # the maximum number of visable fringes
                num_fringes = np.ceil(
                    (width_box[1] - width_box[0]) * baseline.num_pairs / fringe_spacing
                )

                # location on the detector with pathlength diffrence of zero, assuming the mirrors
                # have been alligned to correct for the distance between the combining mirrors
                off_set_fringes = -(baseline.D * np.sin(thetas)) / (
                    2 * np.sin(baseline.beam_angle / 2)
                )

                # find the left most fringe partially or entirely able to reach the detector
                left_fringe = off_set_fringes - (
                    np.ceil(
                        (
                            off_set_fringes
                            - baseline.L * thetas
                            + baseline.W * baseline.num_pairs / 2
                        )
                        / fringe_spacing
                    )
                    * fringe_spacing
                )

                # samples where in the fringe each photon arrives
                rand_cos = (
                    0.5
                    * fringe_spacing
                    * sampler.cosine.rvs(
                        loc=2 * np.pi * left_fringe / fringe_spacing, size=samples_left
                    )
                    / np.pi
                )

                if pure_diffraction:
                    # selects a random uniform position, resulting in no fringe sampling
                    rand_pos = np.random.uniform(
                        width_box[0], width_box[1], size=samples_left
                    )
                else:
                    # selects wich of the visable fringes each photon arrives in
                    rand_pos = rand_cos + (
                        np.random.randint(
                            low=0, high=num_fringes + 1, size=samples_left
                        )
                        * fringe_spacing
                    )

                # let the user know when the fringes would in practice move outside of the FoV
                if np.logical_or(
                    (off_set_fringes < width_box[0]).any(),
                    (off_set_fringes > width_box[1]).any(),
                ):
                    warnings.warn("There are fringe centres outside of the FOV!")

                # boolean for wich y posistions don't fall outside of the allowed range,
                # due to sampling an integer number of fringes and not fractional ammounts
                valid_rand_pos = np.logical_and(
                    rand_pos >= width_box[0], rand_pos <= width_box[1]
                )  # np.repeat(True, samples_left) #

                # map sampled positions to the central diffraction pattern
                # the diffraction pattern is assumed to be multiple patterns
                # next to one another without interfering
                if (baseline.num_pairs % 2) == 0:
                    diffrac_pos = (abs(rand_pos + baseline.L * thetas) % baseline.W) - (
                        baseline.W / 2
                    )  # rand_pos #
                else:
                    diffrac_pos = (
                        abs(rand_pos + baseline.L * thetas + baseline.W / 2)
                        % baseline.W
                    ) - (baseline.W / 2)  # rand_pos #

                # accept reject
                diffraction_value, _ = self.fre_dif(
                    wavelengths, baseline, 1, diffrac_pos
                )
                rand_value = np.random.uniform(
                    height_box[0], height_box[1], size=samples_left
                )
                if pure_fringes:
                    accept_reject = np.repeat(True, samples_left)
                else:
                    accept_reject = rand_value <= diffraction_value

                # keep the accepted y posistions
                y_pos[indices[accept_reject * valid_rand_pos]] = rand_pos[
                    accept_reject * valid_rand_pos
                ]

                # only keep the rejected theta's and wavelengths for the next round of accept reject
                thetas = thetas[np.invert(accept_reject * valid_rand_pos)]
                wavelengths = wavelengths[np.invert(accept_reject * valid_rand_pos)]
                times = times[np.invert(accept_reject * valid_rand_pos)]
                indices = indices[np.invert(accept_reject * valid_rand_pos)]

            # here the sampled photon positions are saved
            actual_pos[photons_in_baseline] = y_pos

        return actual_pos


def generate_event_list(
    src_file, arf_path, rmf_path, exposure_time, output_file="event_list.fits"
):
    # Load spectrum data
    with fits.open(src_file) as hdul:
        spectrum_data = hdul["SPECTRUM"].data
        src_id = hdul["SRC_CAT"].data["SRC_ID"]
        coords = hdul["SRC_CAT"].data["RA"], hdul["SRC_CAT"].data["DEC"]
    hdul.close()

    with fits.open(arf_path) as hdul_arf:
        num_baselines = len(hdul_arf) - 2
        multiplied_spectra = np.zeros(
            (num_baselines, len(src_id), len(spectrum_data["ENERGY"][0]))
        )

        for i in range(num_baselines):
            ext_name = f"SPECRESP_BL{i + 1}"
            arf_data = hdul_arf[ext_name].data

            # Calculate ARF energy midpoints
            arf_energy = (arf_data["ENERG_LO"] + arf_data["ENERG_HI"]) / 2.0
            arf_resp = arf_data[ext_name]
            energy_bin_width = arf_data["ENERG_HI"] - arf_data["ENERG_LO"]

            # Iterate over sources
            for j in range(len(src_id)):
                src_energy = spectrum_data["ENERGY"][j]
                src_flux = spectrum_data["FLUXDENSITY"][j]

                # Interpolate ARF effective area to the source energy grid for multiplication
                arf_resp_interp = np.interp(
                    src_energy, arf_energy, arf_resp, left=0, right=0
                )

                # Multiply flux by effective area and energy bin width to get counts
                res = src_flux * arf_resp_interp * energy_bin_width
                multiplied_spectra[i, j, :] = res
    hdul_arf.close()

    # draw photons from multiplied spectra and create event list
    all_energies = []
    all_baselines = []
    all_times = []
    all_ra = []
    all_dec = []

    for i in range(num_baselines):
        for j in range(len(src_id)):
            # Calculate expected counts per energy bin
            expected_counts_per_bin = multiplied_spectra[i, j, :] * exposure_time

            # Sample actual counts using Poisson statistics
            counts_per_bin = np.random.poisson(expected_counts_per_bin)

            # Get energy values for this source
            energies = spectrum_data["ENERGY"][j]
            # Generate photon events: repeat energy value N times where N is the sampled count
            sampled_energies = np.repeat(energies, counts_per_bin)
            sampled_baselines = np.full(len(sampled_energies), i, dtype=int)
            sampled_times = np.random.uniform(0, exposure_time, len(sampled_energies))

            # Add sky location to sampled photons
            ra_pos = coords[0][j]
            dec_pos = coords[1][j]

            sampled_ra = np.full(len(sampled_energies), ra_pos)
            sampled_dec = np.full(len(sampled_energies), dec_pos)
            all_times.append(sampled_times)
            all_energies.append(sampled_energies)
            all_baselines.append(sampled_baselines)
            all_ra.append(sampled_ra)
            all_dec.append(sampled_dec)

    # Concatenate all events into single arrays
    if all_energies:
        # Concatenate source energies first (true energies)
        true_energies = np.concatenate(all_energies)

        # Load RMF
        with fits.open(rmf_path) as hdul_rmf:
            rmf_data = hdul_rmf["MATRIX"].data
            ebounds_data = hdul_rmf["EBOUNDS"].data
        hdul_rmf.close()

        # RMF energy grid
        f_chan = rmf_data["F_CHAN"]
        n_chan = rmf_data["N_CHAN"]
        matrix = rmf_data["MATRIX"]
        energ_lo = rmf_data["ENERG_LO"]
        energ_hi = rmf_data["ENERG_HI"]

        # EBOUNDS for channel to energy conversion
        channel_min = ebounds_data["E_MIN"]
        channel_max = ebounds_data["E_MAX"]

        final_energies = []

        # Redistribution Loop (simplistic approach for demonstration)
        # Map true energies to RMF energy bins
        # This can be slow for large number of photons, vectorized approaches are complex due to sparse matrix
        # Finding the RMF bin index for each photon
        bin_indices = np.searchsorted(energ_hi, true_energies)

        # Filter out photons outside RMF range
        valid_mask = (bin_indices < len(energ_lo)) & (true_energies >= energ_lo[0])
        bin_indices = bin_indices[valid_mask]

        # Update other arrays to match valid photons
        final_baselines = np.concatenate(all_baselines)[valid_mask]
        final_times = np.concatenate(all_times)[valid_mask]

        # Sample channels
        sampled_channels = []

        for idx in bin_indices:
            # Get matrix row for this energy bin
            m_row = matrix[idx]

            # In FITS RMF, sparse storage:
            # F_CHAN is array of start channels for groups
            # N_CHAN is array of number of channels in each group
            # MATRIX is flat array of probabilities

            if len(m_row) == 0:
                sampled_channels.append(-1)  # No response
                continue

            # Reconstruct full probability array for this energy bin is needed for choice
            # However, usually we can just construct the specific valid channels and probs

            current_f_chan = f_chan[idx]
            current_n_chan = n_chan[idx]

            # If scalar (not variable length array logic in some files), handling depends on file format nuances
            # Assuming standard OGIP FITS variable length arrays

            if np.isscalar(current_f_chan):
                current_f_chan = [current_f_chan]
                current_n_chan = [current_n_chan]

            possible_channels = []
            probabilities = []

            # Offset in the flattened matrix array for the current row
            mat_idx = 0
            for k in range(len(current_f_chan)):
                start = current_f_chan[k]
                count = current_n_chan[k]
                chunk = m_row[mat_idx : mat_idx + count]

                possible_channels.extend(range(start, start + count))
                probabilities.extend(chunk)
                mat_idx += count

            probabilities = np.array(probabilities)

            # Ensure non-negative probabilities (clip small negative values to 0)
            probabilities[probabilities < 0] = 0
            prob_sum = probabilities.sum()

            if prob_sum > 0:
                probabilities = probabilities / prob_sum  # Normalize

                # Sample one channel
                sampled_chan = np.random.choice(possible_channels, p=probabilities)
                sampled_channels.append(sampled_chan)
            else:
                sampled_channels.append(-1)

        sampled_channels = np.array(sampled_channels)

        # Filter out failed encodings
        valid_chan_mask = sampled_channels != -1
        sampled_channels = sampled_channels[valid_chan_mask]
        final_baselines = final_baselines[valid_chan_mask]
        final_times = final_times[valid_chan_mask]

        # Ensure channel indices are within bounds
        valid_bounds = sampled_channels < len(channel_min)
        sampled_channels = sampled_channels[valid_bounds]
        final_baselines = final_baselines[valid_bounds]
        final_times = np.round(final_times[valid_bounds], 3)
        ra_pos = np.concatenate(all_ra)[valid_bounds]
        dec_pos = np.concatenate(all_dec)[valid_bounds]

        final_energies = (
            channel_min[sampled_channels] + channel_max[sampled_channels]
        ) / 2.0

    else:
        final_energies = np.array([])
        final_baselines = np.array([])
        final_times = np.array([])

    # Create a combined event list
    num_events = len(final_energies)
    event_list = np.column_stack(
        (final_baselines, true_energies, final_energies, final_times)
    )

    print(f"Total photons generated: {num_events}")

    # save event list to fits file
    primary_hdu = fits.PrimaryHDU()
    cols = [
        fits.Column(name="BASELINE_ID", format="I", array=event_list[:, 0]),
        fits.Column(name="TRUE_ENERGY", format="E", array=event_list[:, 1]),
        fits.Column(name="DETECTED_ENERGY", format="E", array=event_list[:, 2]),
        fits.Column(name="TIME", format="E", array=event_list[:, 3]),
        fits.Column(name="RA_POS", format="D", array=ra_pos),
        fits.Column(name="DEC_POS", format="D", array=dec_pos),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header.update(
        {
            "EXTNAME": "EVENTS",
            "TELESCOP": "XRI",
            "INSTRUME": "CIS",
            "TTYPE1": "BASELINE_ID",
            "TTYPE2": "TRUE_ENERGY",
            "TTYPE3": "DETECTED_ENERGY",
            "TTYPE4": "TIME",
            "TTYPE5": "RA_POS",
            "TTYPE6": "DEC_POS",
            "TUNIT1": "",
            "TUNIT2": "keV",
            "TUNIT3": "keV",
            "TUNIT4": "s",
            "TUNIT5": "rad",
            "TUNIT6": "rad",
        }
    )
    hdul = fits.HDUList([primary_hdu, hdu])
    hdul.writeto(output_file, overwrite=True)
    hdul.close()
