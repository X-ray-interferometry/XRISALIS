
import re
import math
import warnings
import numpy as np
import scipy.constants as spc
import scipy.interpolate as interp

import matplotlib.pyplot as plt
from astropy.io import fits


class baseline():
    """
    This class defines a single baseline in an interferometer object, and is used as a helper for the interferometer class objects.
    #TODO add more relevant parameters to make this more realistic. In order to fully accurately model an observation,
    this class can be expanded. This would necesarily also include another conceptual shift with
    consequences through the rest of the code, as at the moment the image class represents a collection of all photons that will be detected, 
    which would need to shift to being a collection of photons that could be detected, with the number of input photons likely being much greater
    than the detected photons. 
    """

    def __init__(self, num_pairs, D = None, L = None, W = None, beam_angle = None, F = None,
                 grazing_angle = None, bench_length = None, interferometer = None, mirr_reflec = None):
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
                self.beam_angle = W * 1e-6 / L # W to SI units
            except TypeError:
                raise Exception(r"ERROR: Either define the beam angle ($\theta_b$) or both the beam width ($W$) and the combining length ($L$)!")
        if D is not None:
            self.D = D
        else:
            try:
                self.D = F * self.beam_angle
            except TypeError:
                raise Exception(r"ERROR: Either define the baseline length ($D$) or the effective focal length ($F$)!")
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
                raise Exception(r"ERROR: Either define the beam width ($W$) or the combining length ($L$)!")
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
                self.grazing_angle = np.arctan( self.D / 2 / self.bench_length) / 2
            except (TypeError, AttributeError):
                warnings.warn(r"Warning: grazing angle could not be calculated and is set to None.")
                self.grazing_angle = None

        self.num_pairs = num_pairs

        # define standard detector
        # self.camera = detector(res_E = 0.1, res_t = 1, res_pos = 2, E_range = np.array([1, 7]), pos_range = np.array([-22000, 22000])) #np.array([-1000, 1000])) #np.array([-300, 300])) ## current CMOS

        # initialize the search for the sampled angles in the file names
        float_finder = re.compile(f'.*([0-9]+\\.[0-9]+)')

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
                reflec_data.append(np.loadtxt(item, skiprows = 2).T)

            # save to attribute
            self.mirr_reflec = np.array([angles, reflec_data], dtype = object).T


        # check if input values conform to physical constraints, use math.isclose against binary fraction approximation errors
        if not math.isclose(self.D, self.F * self.beam_angle):
            raise Exception(r"ERROR: The chosen baseline length ($D$), effective focal length ($F$) and beam angle ($\theta_b$) do NOT match!")
        if not math.isclose(self.W, self.beam_angle * self.L):
            raise Exception(r"ERROR: The chosen beam width ($W$), combining length ($L$) and beam angle ($\theta_b$) do NOT match!")
        if type(self.num_pairs) is not int or self.num_pairs <= 0:
            raise Exception(r"ERROR: The number of pairs must be an integer and at least one!")

        
class interferometer():
    """ 
    Class defining a hypothetical x-ray interferometer.
    It contains the code needed to generate the interferometer and adapt some of its characteristics afterwards.
    """

    def __init__(self, time_step = 1, wobbler = None, wobble_I = 0., wobble_c = None, wobble_file = '', 
                    roller = None, roll_speed = 0., roll_stop_t = 0., roll_stop_a = 0., roll_init = 0,
                    max_ob_length = None):
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
        pointing[1:, :2] = pointing[:-1, :2] + np.random.normal(0, self.wobble_I, size=(len(pointing[:, 0]) - 1, 2)) * 2 * np.pi / (3600 * 360)
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
        pointing[:, :2] = np.genfromtxt(self.wobble_file, np.float64, delimiter=',')
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
        pointing[:, 2] = (np.arange(pointing[:, 2].size) * self.roll_speed * self.time_step) + self.roll_init
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

    def add_baseline(self, num_pairs, D = None, L = None, W = None, beam_angle = None, F = None,
                     grazing_angle = None, bench_length = None, interferometer = None, mirr_mater = None):
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
        self.baselines.append(baseline(num_pairs, D, L, W, beam_angle, F,
                                       grazing_angle, bench_length, interferometer,
                                       mirr_mater))

    def add_willingale_baseline(self, D):
        """
        Function that adds a baseline with the parameters described in Willingale (2004) to the interferometer object.
        Call this function multiple times to construct a full interferometer capable of actually observing images.
        Without these, no photons can be measured.
        """
        
        self.baselines.append(baseline(num_pairs = 30, D = D, L = 10, W = 300, bench_length= 7))

    def clear_baselines(self):
        self.baselines.clear()



def plot_arf(arf_file, baseline_mode = False):
    """ 
    Function that plots the effective area from a given arf file. 
    Parameters:
    arf_file (string) = Path to the arf file to plot.\n
    baseline_mode (bool) = Whether to plot in baseline mode (effective area per baseline) or not (total effective area). Default is False.\n
    """

    arf_data = fits.open(arf_file)

    energies = arf_data[1].data['ENERG_LO'] + (arf_data[1].data['ENERG_HI'] - arf_data[1].data['ENERG_LO']) / 2
    eff_area = arf_data[1].data['SPECRESP']

    # there are additional baseline extensions to arf files, which store the effective area per baseline, named 'SPECRESP_BL{i}' where i starts from 1 and is the baseline ID, find maximum i and plot all arfs per baseline if in baseline mode

    plt.figure(figsize=(7,5))
    if baseline_mode:
        i = 1
        while 'SPECRESP_BL' + str(i) in arf_data:
            eff_area_bl = arf_data['SPECRESP_BL' + str(i)].data['SPECRESP_BL' + str(i)]
            # get the baseline in meters from the baseline extension header
            baseline_length = arf_data['SPECRESP_BL' + str(i)].header.get('BASELINE', None)
            plt.plot(energies, eff_area_bl, label = f'Baseline {i} ({baseline_length:.2f} m)', color='mediumvioletred', linewidth=1, alpha= 1/i )
            i += 1
    plt.plot(energies, eff_area, label = 'Total Effective Area', color='mediumvioletred', linewidth=2)
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