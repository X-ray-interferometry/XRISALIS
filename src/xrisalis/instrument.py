import re
import math
import warnings
import numpy as np

import matplotlib.pyplot as plt
from astropy.io import fits

# class detector():
#     """
#     Class defining a hypothetical x-ray detector.
#     It contains the code needed to generate the interferometer and adapt some of its characteristics afterwards.
#     """

#     def __init__(self, res_E, res_t, res_pos, E_range, pos_range,
#                  pos_noise = 0., energy_noise = 0., t_noise = 0.,
#                  quant_eff = r"../XRImulator/Models/detector_qe/Si_9p5_um_transmission_data.txt",
#                  response_matrix = None):
#         """
#         This function is the main function that takes the 'real' photons at the camera and
#         converts them to detector output data as if the detector had just detected and measured those photons.
#         It models the detector ranges, resolutions and noise for time, posistion and energie measurements
#         whether they are absorbed along the way, how much noise there is.

#         TODO it can be adopted to include more realistic models of detectors; and or a seperate background class;
#         and or energy consumption and readout times; and or more.

#         Parameters:\n
#         res_E (float) = Energy resolution of CCD's in instrument (in KeV)\n
#         res_t (float) = Time resolution of CCD's in instrument (seconds)\n
#         res_pos (float) = Position resolution of CCD's in instrument (in micrometers)\n

#         E_range (array-like of floats) = Range of energies that can be recorded (in KeV)\n
#     	pos_range (array-like of floats) = Range of positions that can be recorded (in micrometers)\n #set by detector surface area

#         pos_noise (float) = Noise value in micrometers used as sigma in normal distribution around 'true' position. Default 0. means no noise.\n
#         energy_noise (float) = Noise value used as percentage of 'true' energy to determine sigma in normal distribution. Default 0. means no noise.\n
#         t_noise (float) = Noise value in seconds used as sigma in normal distribution around 'true' time of arrival. Default 0. means no noise.\n

#         quant_eff (string) = A file name, including the path. The file contains the quantum efficiancy of the detector in the form (1 - QE) per energy.\n
#         response_matrix (string) = A file name, including the path. The file contains the response_matrix of the detector.\n
#         """

#         # Different resolutions, with energy, time and pixel size.
#         self.res_E = res_E * 1e3 * spc.eV
#         self.res_t = res_t
#         self.res_pos = res_pos * 1e-6

#         # energy and positional ranges of the detector
#         self.E_range = E_range * 1e3 * spc.eV
#         self.pos_range = pos_range * 1e-6

#         # Useful shorthand
#         self.pos_noise = pos_noise * 1e-6
#         self.energy_noise = energy_noise * spc.eV * 1e3
#         self.t_noise = t_noise

#         # loading the quantum efficiancy from a file with file format from https://henke.lbl.gov/
#         self.quant_eff = np.loadtxt(quant_eff, skiprows = 2)

#         # TODO: add the reading in of the response matrix of the detector
#         self.response_matrix = response_matrix

#         # add noise to the real energies, positions and arival times TODO: this will later be done by response matrices
#         # self.noise_photon_energies(instrument, instrument_data)
#         # self.noise_photon_pos(instrument, instrument_data)
#         # self.noise_photon_toa(instrument_data)

#         # pixelise the noised data TODO: this will later be done by response matrices
#         # self.discretize_E(instrument)
#         # self.discretize_t(instrument)
#         # self.discretize_pos(instrument)

#     def noise_photon_energies(self, instrument_data, where):
#         """
#         This function is a helper function for process_image that specifically processes the energies that photons have and
#         how the instrument records them.
#         """
#         if self.energy_noise > 0.:
#             # % is for forcing it to be impossible for photons to be measured above or below energy range, while keeping random distribution
#             # If you want to avoid high energies bleeding over in the case of for example an emission line you want to image,
#             # simply set the energy range too big to have this contamination.
#             instrument_data.energies[where] = instrument_data.energies[where] + np.random.normal(0, self.energy_noise, instrument_data.energies[where].size)
#                                                 # - self.E_range[0])
#                                                 #     % (self.E_range[1] - self.E_range[0])
#                                                 #     + self.E_range[0])
#             instrument_data.energies[where][instrument_data.energies[where] < self.E_range[0]] = self.E_range[0]
#             instrument_data.energies[where][instrument_data.energies[where] > self.E_range[1]] = self.E_range[1]
#         return instrument_data

#     def noise_photon_toa(self, instrument_data, where):
#         """
#         This function is a helper function for process_image that specifically processes the times at which photons arrive and
#         how the instrument records them.
#         """
#         if self.t_noise > 0.:
#             # % is for forcing it to be impossible for photons to arrive late or early, while keeping random distribution
#             instrument_data.toa[where] = np.random.normal(instrument_data.toa[where], self.t_noise, instrument_data.toa[where].size)
#                                                         #    % np.max(instrument_data.toa[where]))
#         return instrument_data

#     def noise_photon_pos(self, instrument_data, where):
#         """
#         This function processes the positions at which photons arrive and
#         how the instrument records them.
#         """

#         # Noises up the data
#         if self.pos_noise > 0.:
#             instrument_data.pos[where] = instrument_data.pos[where] + np.random.normal(0, self.pos_noise, instrument_data.pos[where].size)# - self.pos_range[0])
#                                             #  % (self.pos_range[1] - self.pos_range[0])
#                                             #  + self.pos_range[0])
#             instrument_data.pos[where][instrument_data.pos[where] < self.pos_range[0]] = self.pos_range[0]
#             instrument_data.pos[where][instrument_data.pos[where] > self.pos_range[1]] = self.pos_range[1]

#         return instrument_data

#     def discretize_E(self, instrument_data, where):
#         """
#         Method that discretizes energies of incoming photons into energy channels.
#         Adds an array of these locations stored to the class under the name self.discrete_E.
#         """
#         instrument_data.energies[where] = (instrument_data.energies[where] - self.E_range[0]) // self.res_E

#     def channel_to_E(self, instrument_data, where):
#         """ Method that turns discretized energies into the energies at the center of their respective channels. """
#         instrument_data.energies[where] = (instrument_data.energies[where] + .5) * self.res_E + self.E_range[0]

#     def discretize_pos(self, instrument_data, where):
#         """
#         Method that discretizes positions of incoming photons into pixel positions.
#         Adds an array of these locations stored to the class under the name self.discrete_pos.
#         """
#         instrument_data.pos[where] = (instrument_data.pos[where] - self.pos_range[0]) // self.res_pos

#     def pixel_to_pos(self, instrument_data, where):
#         """ Method that turns discretized positions into the positions at the center of their respective pixels. """
#         instrument_data.pos[where] = (instrument_data.pos[where] + .5) * self.res_pos + self.pos_range[0]

#     def discretize_t(self, instrument_data, where):
#         """
#         Method that discretizes times of arrival of incoming photons into time steps since start of observation.
#         Adds an array of these times stored to the class under the name self.discrete_t.
#         """
#         instrument_data.toa[where] = ((instrument_data.toa[where] - instrument_data.toa[0]) // self.res_t)#.astype(int)

#     def tstep_to_t(self, instrument_data, where):
#         """ Method that turns discretized time steps into the times at the center of their respective steps. """
#         instrument_data.toa[where] = (instrument_data.toa[where] + .5) * self.res_t

#     def add_detector_effects(self, instrument_data, where):
#         """ Applies the detector effects on the exact photon properties

#             Parameters:

#             instrument_data (interferometer_data object) = the exact photon properties.\n
#             where (list): list of booleans on which photons to apply the effects.
#         """

#         instrument_data = self.noise_photon_energies(instrument_data, where)
#         instrument_data = self.noise_photon_toa(instrument_data, where)
#         instrument_data = self.noise_photon_pos(instrument_data, where)
#         self.discretize_E(instrument_data, where)
#         self.channel_to_E(instrument_data, where)
#         self.discretize_pos(instrument_data, where)
#         self.pixel_to_pos(instrument_data, where)
#         self.discretize_t(instrument_data, where)
#         self.tstep_to_t(instrument_data, where)

#         return instrument_data


class baseline:
    """
    This class defines a single baseline in an interferometer object, and is used as a helper for the interferometer class objects.
    #TODO add more relevant parameters to make this more realistic. In order to fully accurately model an observation,
    this class can be expanded. This would necesarily also include another conceptual shift with
    consequences through the rest of the code, as at the moment the image class represents a collection of all photons that will be detected,
    which would need to shift to being a collection of photons that could be detected, with the number of input photons likely being much greater
    than the detected photons.
    """

    def __init__(
        self,
        num_pairs,
        D=None,
        L=None,
        W=None,
        beam_angle=None,
        F=None,
        grazing_angle=None,
        bench_length=None,
        interferometer=None,
        mirr_reflec=None,
    ):
        """
        Function that generates a single x-ray interferometer baseline according to given specifications.


        Parameters:\n
        num_pairs (int) = Number of slit-gap pairs in the slatted mirror\n
        D (float) = Baseline of the interferometer (in meters)\n
        L (float) = Length from last mirror to CCD surface (in meters)\n
        W (float) = Incident photon beam width (in micrometers)\n # set by projected slat width
        beam_angle (float) = Angle between the two beams at the detector (in radians)\n
        F (float) = Effective focal length of interferometer (in meters)\n
        grazing_angle (float) = Angle of the mirrors with respect to the beam (in radians)\n
        bench_length (float) = Length of the optical bench (in meters)\n
        interferometer (class interferometer) = Interferometer the baseline is a part of\n
        mirr_reflec (list) = The refrectivity of the mirror material given certain energies and angles. Sampled angle must be in file name.
        """

        # Converting all input parameters into self.parameters in SI units, either by direct assignment or by calculation.
        # See Willingale 2004 for all equations if intrested.
        if beam_angle is not None:
            self.beam_angle = beam_angle
        else:
            try:
                self.beam_angle = W * 1e-6 / L  # W to SI units
            except TypeError:
                raise Exception(
                    r"ERROR: Either define the beam angle ($\theta_b$) or both the beam width ($W$) and the combining length ($L$)!"
                )
        if D is not None:
            self.D = D
        else:
            try:
                self.D = F * self.beam_angle
            except TypeError:
                raise Exception(
                    r"ERROR: Either define the baseline length ($D$) or the effective focal length ($F$)!"
                )
        if F is not None:
            self.F = F
        else:
            self.F = self.D / self.beam_angle
        if W is not None:
            self.W = W * 1e-6
        else:
            try:
                self.W = self.beam_angle * L
            except TypeError:
                raise Exception(
                    r"ERROR: Either define the beam width ($W$) or the combining length ($L$)!"
                )
        if L is not None:
            self.L = L
        else:
            self.L = self.W / self.beam_angle
        if bench_length is not None:
            self.bench_length = bench_length
        # else:
        #     try:
        #         self.bench_length = 0.5 * self.D / np.tan(2 * grazing_angle)
        #     except TypeError:
        #         warnings.warn(r"Warning: if you want to check the length contraints, either define the length of the interferometer arm projected onto the optical axis ($B$)"+
        #                       r" or the grazing angle ($\theta_g$)!")
        if grazing_angle is not None:
            self.grazing_angle = grazing_angle
        else:
            try:
                self.grazing_angle = np.arctan(self.D / 2 / self.bench_length) / 2
            except (TypeError, AttributeError):
                warnings.warn(
                    r"Warning: grazing angle could not be calculated and is set to None."
                )
                self.grazing_angle = None

        self.num_pairs = num_pairs

        # define standard detector
        # self.camera = detector(res_E = 0.1, res_t = 1, res_pos = 2, E_range = np.array([1, 7]), pos_range = np.array([-22000, 22000])) #np.array([-1000, 1000])) #np.array([-300, 300])) ## current CMOS

        # initialize the search for the sampled angles in the file names
        float_finder = re.compile(r".*([0-9]+\.[0-9]+)")

        angles = []
        reflec_data = []

        # no mirror reflectivity is given, initialize attribute to avoid attribute errors elsewhere
        if mirr_reflec is None:
            self.mirr_reflec = mirr_reflec

        else:
            # loop through all sampled angles
            for item in mirr_reflec:
                # find and save the sampled angle
                angles.append(float(float_finder.search(item).group(1)))

                # read and save the mirror reflectivity per sampled energy, assume https://henke.lbl.gov/ file format
                reflec_data.append(np.loadtxt(item, skiprows=2).T)

            # save to attribute
            self.mirr_reflec = np.array([angles, reflec_data], dtype=object).T

        # shows the mirror reflectivity
        # for angle in self.mirr_reflec:
        #     plt.plot(angle[1][0] / 1000, angle[1][1], label = r"$\theta_{\mathrm{g}} = $" + str(angle[0]) + r"$^{\circ}$")

        # plt.ylabel("Reflectivity")
        # plt.xlabel("Energy (keV)")
        # plt.xlim(1, 7)
        # plt.ylim(0, 1)
        # plt.legend()
        # # plt.savefig("reflectivity.pdf")
        # plt.show()

        # check if input values conform to physical constraints, use math.isclose against binary fraction approximation errors
        if not math.isclose(self.D, self.F * self.beam_angle):
            raise Exception(
                r"ERROR: The chosen baseline length ($D$), effective focal length ($F$) and beam angle ($\theta_b$) do NOT match!"
            )
        if not math.isclose(self.W, self.beam_angle * self.L):
            raise Exception(
                r"ERROR: The chosen beam width ($W$), combining length ($L$) and beam angle ($\theta_b$) do NOT match!"
            )
        if type(self.num_pairs) is not int or self.num_pairs <= 0:
            raise Exception(
                r"ERROR: The number of pairs must be an integer and at least one!"
            )

        # check if the length is within limits, only when possible
        # try:
        #     if interferometer.max_ob_length < self.L + self.distance_M1_4() + self.bench_length:
        #         warnings.warn("Warning: the chosen obtical bench exeeds the maximum length!")
        # except (TypeError, AttributeError):
        #     warnings.warn("Warning: missing information to check the length restrictions!")

    # def add_custom_detector(self, res_E, res_t, res_pos, E_range, pos_range, pos_noise = 0., E_noise = 0., t_noise = 0.,
    #                         quant_eff = r"CModels/Detector QE/Si_9p5_um_transmission_data.txt",
    #                         response_matrix = None):
    #     """ Allows for defining a custom detector.

    #         Parameters:\n
    #         res_E (float) = Energy resolution of CCD's in instrument (in KeV)\n
    #         res_t (float) = Time resolution of CCD's in instrument (seconds)\n
    #         res_pos (float) = Position resolution of CCD's in instrument (in micrometers)\n

    #         E_range (array-like of floats) = Range of energies that can be recorded (in KeV)\n
    #         pos_range (array-like of floats) = Range of positions that can be recorded (in micrometers)\n #set by detector surface area

    #         pos_noise (float) = Noise value in micrometers used as sigma in normal distribution around 'true' position. Default 0. means no noise.\n
    #         energy_noise (float) = Noise value used as percentage of 'true' energy to determine sigma in normal distribution. Default 0. means no noise.\n
    #         t_noise (float) = Noise value in seconds used as sigma in normal distribution around 'true' time of arrival. Default 0. means no noise.\n

    #         quant_eff (string) = A file name, including the path. The file contains the quantum efficiancy of the detector in the form (1 - QE) per energy.\n
    #         response_matrix (string) = A file name, including the path. The file contains the response_matrix of the detector.\n

    #     """
    #     self.camera = detector(res_E, res_t, res_pos, E_range, pos_range, pos_noise, E_noise, t_noise, quant_eff, response_matrix)

    # def path_difference_change(self, times):
    #     """Add function calls that calculate the path length difference changing over time due to different effects"""

    #     # placeholder value of 0 m
    #     return 0 * times

    # def eff_area(self, photon_energies):

    #     # return absolute area when no reflectivity information is given,
    #     # asumes perfect reflectivity and camera quantum efficiency
    #     if self.mirr_reflec == None:
    #         # assuming square camera / detector
    #         apperature_area = np.power(self.camera.pos_range[1] - self.camera.pos_range[0], 2)
    #         return np.repeat(apperature_area, photon_energies.size)

    #     else:
    #         values = []
    #         sampled_energies = []
    #         sampled_angles = []

    #         # looping over all sampled reflection angle
    #         for item in self.mirr_reflec:

    #             # retrieving sampled energies and their respected reflectivity at the sampled angle
    #             energies, reflectivity = item[1]
    #             values.extend(reflectivity)

    #             # assume energies are given in eV, turn into J
    #             sampled_energies.extend(energies * spc.eV)

    #             # assume angles are given in degries, turn into radians
    #             sampled_angles.extend(np.repeat(item[0] * np.pi / 180, energies.size))

    #         points = np.array([sampled_energies, sampled_angles]).T

    #         """TODO: Below is slow"""
    #         mirr_reflec = interp.griddata(points, values, (photon_energies, np.repeat(self.grazing_angle, photon_energies.size)), method='nearest')

    #         # assuming square camera / detector
    #         apperature_area = np.power(self.camera.pos_range[1] - self.camera.pos_range[0], 2)

    #         # retrieve and interpolate the quantum efficiancy of the camera / detector
    #         cam_quant_eff = self.camera.quant_eff.T

    #         """Testing plots"""
    #         # plt.plot(cam_quant_eff[0] / 1000, 1 - cam_quant_eff[1])
    #         # plt.ylabel("Quantum efficiency")
    #         # plt.xlabel("Energy (keV)")
    #         # plt.xlim(1, 7)
    #         # plt.ylim(0, 1)
    #         # plt.savefig("quant_eff.pdf")
    #         # plt.show()

    #         cam_eff_area = interp.interp1d(cam_quant_eff[0] * spc.eV, 1 - cam_quant_eff[1])(photon_energies)

    #         # equation in 4.3 Mirror quality and effective area from Phil Uttley et al. (2020)
    #         mirr_eff_area = (1/12) * apperature_area * np.power(mirr_reflec, 2)

    #         return mirr_eff_area * cam_eff_area


class interferometer:
    """
    Class defining a hypothetical x-ray interferometer.
    It contains the code needed to generate the interferometer and adapt some of its characteristics afterwards.
    """

    def __init__(
        self,
        time_step=1,
        wobbler=None,
        wobble_I=0.0,
        wobble_c=None,
        wobble_file="",
        roller=None,
        roll_speed=0.0,
        roll_stop_t=0.0,
        roll_stop_a=0.0,
        roll_init=0,
        max_ob_length=None,
    ):
        """
        Function that generates a virtual x-ray interferometer according to given specifications.

        Parameters:\n
        time_step (float) = arbitrary time steps in the simulator, to be based on fluxes (s)\n

        wobbler (function) = Function to use to simmulate wobble in observation (possibly not relevant here)\n
        wobble_I (float) = Intensity of wobble effect, used as sigma in normally distributed random walk steps. Default is 0, which means no wobble. (in arcsec)\n
        wobble_c (function) = Function to use to correct for spacecraft wobble in observation (possibly not relevant here)\n
        wobble_file (file) = File containing spacecraft wobble pointing positions.\n

        roller (function) = Function to use to simulate the spacecraft rolling. Options are 'smooth_roll' and 'discrete_roll'.\n
        roll_speed (float) = Indicator for how quickly spacecraft rolls around. Default is 0, meaning no roll. (in rad/sec)\n
        roll_stop_t (float) = Indicator for how long spacecraft rests at specific roll if using 'discrete_roll'. Default is 0, meaning it doesn't stop. (in seconds)\n
        roll_stop_a (float) = Indicator for at what angle increments spacecraft rests at if using 'discrete_roll'. Default is 0, meaning it doesn't stop. (in rads)\n
        roll_init (float) = Initial roll angle. (in radians)\n

        max_ob_length (float) = Maximum length of the optical bench of the spacecraft. (in m)\n
        """

        self.baselines = []

        self.time_step = time_step

        self.wobbler = wobbler
        self.wobble_I = wobble_I
        self.wobble_c = wobble_c
        self.wobble_file = wobble_file

        self.roller = roller
        self.roll_speed = roll_speed
        self.roll_stop_t = roll_stop_t
        self.roll_stop_a = roll_stop_a
        self.roll_init = roll_init

        self.max_ob_length = max_ob_length

    def random_wobble(self, pointing):
        """
        Function that adds 'wobble' to the spacecraft, slightly offsetting its pointing every timestep.
        It models wobble as a random walk with a given intensity that is used as the sigma for a normally distributed
        step size in both the pitch and yaw directions.

        Parameters:

        Instrument (interferometer class object): instrument to offset.\n

        Returns:

        pointing (array): That same, but now with wobble data.\n
        """
        pointing[1:, :2] = pointing[:-1, :2] + np.random.normal(
            0, self.wobble_I, size=(len(pointing[:, 0]) - 1, 2)
        ) * 2 * np.pi / (3600 * 360)
        return pointing

    def file_wobble(self, pointing):
        """
        Function that adds 'wobble' to the spacecraft, slightly offsetting its pointing every timestep.
        This function uses an input file in a csv format (with ',' as delimiter) to read out pointing data,
        probably generated with a different simulator.
        #TODO This function is mostly a placeholder, to be replaced later to adapt to the actual format this data
        will take. This is only one way it could look, but it should how to structure an eventual replacement for
        whoever wants to adapt the code.

        Parameters:

        Instrument (interferometer class object): instrument to offset.\n

        Returns:

        pointing (array): That same, but now with wobble data.\n
        """
        pointing[:, :2] = np.genfromtxt(self.wobble_file, np.float64, delimiter=",")
        return pointing

    def smooth_roller(self, pointing):
        """
        Function that generates the roll portion of the pointing data for the instrument.
        This function is used for a continuous model of rolling the instrument, with a predefined roll
        velocity.

        Parameters:

        pointing (array): 3d array of pointing angles as deviations from observation start for every observational timestep.

        Returns:

        pointing (array): That same, but now with roll data.
        """

        self.roll_speed = np.pi / pointing[:, 2].size
        pointing[:, 2] = (
            np.arange(pointing[:, 2].size) * self.roll_speed * self.time_step
        ) + self.roll_init
        return pointing

    def discrete_roller(self, pointing):
        """
        Function that generates the roll portion of the pointing data for the instrument.
        This function is used for a discrete model of rolling the instrument, with starts and stops
        at specified roll angle intervals.

        Parameters:

        pointing (array): 3d array of pointing angles as deviations from observation start for every observational timestep.

        Returns:

        pointing (array): That same, but now with roll data.
        """

        # Calculates the stopping interval in timestep units
        time_to_move = self.roll_stop_t
        # The angle over which to move after the stopping interval
        angle_to_move = self.roll_stop_a

        for i in range(1, pointing[:, 2].size):
            if (i * self.time_step) > time_to_move:
                pointing[i, 2] = angle_to_move
            else:
                pointing[i, 2] = 0

        # # Calculates the stopping interval in timestep units
        # time_to_move = self.roll_stop_t // self.time_step
        # # The angle over which to move after the stopping interval
        # angle_to_move = self.roll_stop_a

        # for i in pointing[:, 2]:
        #     t_to_move = i - time_to_move

        #     if t_to_move > 0.:
        #         pointing[i, 2] = pointing[i - 1, 2] + self.roll_speed * self.time_step
        #     else:
        #         pointing[i, 2] = pointing[i - 1, 2]

        #     # Defining the next timestep to move at and angle to move to.
        #     if pointing[i, 2] > angle_to_move:
        #         angle_to_move += self.roll_stop_a
        #         time_to_move += self.roll_stop_t // self.time_step

        return pointing

    def gen_pointing(self, t_exp):
        """
        This function generates a 3d pointing vector for each time step in an observation. It consists of
        three angles, the pitch, yaw and roll. The first two are linked and generated together by the wobbler
        function, while the roll is fundamentally different and thus generated differently. If no wobbler or
        roller are given, the corresponding pointing values will be zero, indicating stillness.
        """
        pointing = np.zeros((t_exp + 2, 3))

        # These try except statements are there for the case that no roller or wobbler are given.
        try:
            pointing = self.roller(self, pointing)
        except Exception:
            pass

        return pointing

    def add_baseline(
        self,
        num_pairs,
        D=None,
        L=None,
        W=None,
        beam_angle=None,
        F=None,
        grazing_angle=None,
        bench_length=None,
        interferometer=None,
        mirr_mater=None,
    ):
        """
        Function that adds a baseline of given parameters to the interferometer object. Call this function multiple times to
        construct a full interferometer capable of actually observing images. Without these, no photons can be measured.

        Parameters:
        num_pairs (int) = Number of slit-gap pairs in the slatted mirror\n
        D (float) = Baseline of the interferometer (in meters)\n
        L (float) = Length from last mirror to CCD surface (in meters)\n
        W (float) = Incident photon beam width (in micrometers)\n # set by projected slat width
        beam_angle (float) = Angle between the two beams at the detector (in radians)\n
        F (float) = Effective focal length of interferometer (in meters)\n
        grazing_angle (float) = Angle of the mirrors with respect to the beam (in radians)\n
        bench_length (float) = Length of the optical bench (in meters)\n
        interferometer (class interferometer) = Interferometer the baseline is a part of\n
        mirr_mater (float) = The refrective index of the mirror material\n
        """
        self.baselines.append(
            baseline(
                num_pairs,
                D,
                L,
                W,
                beam_angle,
                F,
                grazing_angle,
                bench_length,
                interferometer,
                mirr_mater,
            )
        )

    def add_willingale_baseline(self, D):
        """
        Function that adds a baseline with the parameters described in Willingale (2004) to the interferometer object.
        Call this function multiple times to construct a full interferometer capable of actually observing images.
        Without these, no photons can be measured.
        """

        self.baselines.append(baseline(num_pairs=30, D=D, L=10, W=300, bench_length=7))

    def clear_baselines(self):
        self.baselines.clear()


def plot_arf(arf_file, baseline_mode=False):
    """
    Function that plots the effective area from a given arf file.
    Parameters:
    arf_file (string) = Path to the arf file to plot.\n
    baseline_mode (bool) = Whether to plot in baseline mode (effective area per baseline) or not (total effective area). Default is False.\n
    """

    arf_data = fits.open(arf_file)

    energies = (
        arf_data[1].data["ENERG_LO"]
        + (arf_data[1].data["ENERG_HI"] - arf_data[1].data["ENERG_LO"]) / 2
    )
    eff_area = arf_data[1].data["SPECRESP"]

    # there are additional baseline extensions to arf files, which store the effective area per baseline, named 'SPECRESP_BL{i}' where i starts from 1 and is the baseline ID, find maximum i and plot all arfs per baseline if in baseline mode

    plt.figure(figsize=(7, 5))
    if baseline_mode:
        i = 1
        while "SPECRESP_BL" + str(i) in arf_data:
            eff_area_bl = arf_data["SPECRESP_BL" + str(i)].data["SPECRESP_BL" + str(i)]
            # get the baseline in meters from the baseline extension header
            baseline_length = arf_data["SPECRESP_BL" + str(i)].header.get(
                "BASELINE", None
            )
            plt.plot(
                energies,
                eff_area_bl,
                label=f"Baseline {i} ({baseline_length:.2f} m)",
                color="mediumvioletred",
                linewidth=1,
                alpha=1 / i,
            )
            i += 1
    plt.plot(
        energies,
        eff_area,
        label="Total Effective Area",
        color="mediumvioletred",
        linewidth=2,
    )
    plt.xlabel("Energy (keV)")
    plt.ylabel("Effective area (cm$^2$)")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(np.min(energies), np.max(energies))
    plt.legend(fontsize=12, loc="best")
    if baseline_mode:
        plt.ylim(1e-4, np.max(eff_area) * 2)
    else:
        plt.ylim(1e2, np.max(eff_area) * 1.1)
    plt.grid("both")
    plt.show()
