"""
A class to do useful fiber and photonic lantern things, such as mode finding, coupling, etc.

This class uses
ofiber https://ofiber.readthedocs.io
polarTransform https://polartransform.readthedocs.io/en/latest/getting-started.html
"""
import numpy as np
import ofiber
import matplotlib.pyplot as plt
import polarTransform
# from viewRSoftData import *
from scipy import ndimage
from pathlib import Path

class LanternFiber:
    def __init__(self, 
                 n_core=None, 
                 n_cladding=None, 
                 core_radius=None, 
                 wavelength=None, 
                 nmodes=19,
                 nwgs=19,
                 datadir='./'):

        self.n_core = n_core
        self.n_cladding = n_cladding
        self.core_radius = core_radius
        self.wavelength = wavelength
        self.allmodes_b = None
        self.allmodes_l = None
        self.allmodes_m = None
        self.nmodes = nmodes
        self.nwgs = nwgs
        self.max_r = None
        self.datadir = datadir

        self.all_smpowers = []
        self.all_mmpowers = []
        self.all_mmphases = []
        self.all_smphases = []
        self.Cmat = None # Transfer matrix
        self.Imat = None # Intensity-output matrix
        self.out_field_ampl = []
        self.out_field_phase = []
        self.wg_posns = None
        self.microns_per_pixel = None
        self.npix = None
        self.input_field = None
        self.allmodefields_rsoftorder = None
        self.all_runallbats = []
        self.all_hyperbats = []
        self.all_indiv_commands = []
        self.all_wls = None
        self.allBatfileNames = []

        if n_core is not None:
            self.NA = ofiber.numerical_aperture(n_core, n_cladding)
            self.V = ofiber.V_parameter(core_radius, self.NA, wavelength)

        # If the order of Rsoft monitor objects does not match the conventional waveguide order,
        # specify order here. Using rsoft numbering (so starts at 1).
        self.monitor_order = [10, 9, 14, 15, 11, 6, 5, 4, 3, 8, 13, 18, 19, 16, 17, 12, 7, 2, 1]

        # Specify mode indices
        self.LP_modes = np.array([[0,1],
                             [0,2],
                             [0,3],
                             [1,1],
                             [-1,1],
                             [1,2],
                             [-1,2],
                             [2,1],
                             [-2,1],
                             [2,2],
                             [-2,2],
                             [3,1],
                             [-3,1],
                             [3,2],
                             [-3,2],
                             [4,1],
                             [-4,1],
                             [5,1],
                             [-5,1]
                             ])

        # Make text mode labels
        modelabels = []
        for k in range(self.nmodes):
            if k < 3:  # Assumes first 3 modes are LP0x modes
                label = 'LP%d%d' % (self.LP_modes[k, 0], self.LP_modes[k, 1])
            else:
                if self.LP_modes[k, 0] > 0:
                    suf = 'a'
                else:
                    suf = 'b'
                label = 'LP%d%d' % (np.abs(self.LP_modes[k, 0]), self.LP_modes[k, 1]) + suf
            modelabels.append(label)
        self.modelabels = modelabels


    def find_fiber_modes(self,
                         max_l=100, 
                         return_n_unique=False, 
                         verbose=True):
        """
        Finds LP modes for the specified fiber.

        Input:
            max_l (int): The maximum number of l modes to find
            return_n_unique (bool):
            verbose (bool): If True, display debugging messages
        ----------
        max_l
            Maximum number of l modes to find (can be arbitrarily large)
        """

        # Compute numerical aperture of optic fiber, the sine of the largest angle an incident ray can have for total internal reflectance in the core.
        self.NA = ofiber.numerical_aperture(self.n_core, self.n_cladding)

        # Computes the V number that determines the number of modes the optical fiber can guide
        self.V = ofiber.V_parameter(self.core_radius, self.NA, self.wavelength)

        # Initialize the modes lists
        allmodes_b = []
        allmodes_l = []
        allmodes_m = []

        # Find modes until the max is found or no more LP modes found
        for l in range(max_l):
            cur_b = ofiber.LP_mode_values(self.V, l)
            if len(cur_b) == 0:
                break
            else:
                allmodes_b.extend(cur_b)
                ls = (np.ones_like(cur_b)) * l
                allmodes_l.extend(ls.astype(int))
                ms = np.arange(len(cur_b))+1
                allmodes_m.extend(ms)

        allmodes_b = np.asarray(allmodes_b)
        nLPmodes = len(allmodes_b)
        # print('Total number of LP modes found: %d' % nLPmodes)
        l = np.asarray(allmodes_l)
        total_unique_modes = len(np.where(l == 0)[0]) + len(np.where(l > 0)[0])*2
        if verbose:
            print('Total number of unique modes found: %d' % total_unique_modes)
        self.allmodes_b = allmodes_b
        self.allmodes_l = allmodes_l
        self.allmodes_m = allmodes_m
        self.nLPmodes = nLPmodes

        # ADDED - HACK?
        self.nmodes = total_unique_modes
        if return_n_unique:
            return total_unique_modes


    def make_fiber_modes(
        self,
        max_r=2, 
        npix=100, 
        zlim=0.04, 
        show_plots=False,
        normtosum=True, 
        rotate_mode_angle=None):
        """
        Calculate the LP mode fields, and store as polar and cartesian amplitude maps

        Parameters
        ----------
        max_r
            Maximum radius to calculate mode field, where r=1 is the core diameter
        npix
            Half-width of mode field calculation in pixels
        zlim
            Maximum value to plot
        show_plots : bool
            Whether to produce a plot for each mode
        normtosum : bool
            If True, normalise each mode field so summed power = 1
        """

        r = np.linspace(0, max_r, npix) # Radial positions, normalised so core_radius = 1
        self.max_r = max_r
        self.npix = npix
        self.allmodefields_cos_polar = []
        self.allmodefields_cos_cart = []
        self.allmodefields_sin_polar = []
        self.allmodefields_sin_cart = []
        self.allmodefields_rsoftorder = []

        array_size_microns = self.max_r * self.core_radius * 2
        self.microns_per_pixel = array_size_microns / (npix*2)

        for mode_to_calc in range(self.nLPmodes):
            field_1d = ofiber.LP_radial_field(self.V, self.allmodes_b[mode_to_calc],
                                              self.allmodes_l[mode_to_calc], r)
            #TODO - LP02 comes out with core being pi phase and ring being 0 phase... investigate this.

            phivals = np.linspace(0, 2*np.pi, npix)
            phi_cos = np.cos(self.allmodes_l[mode_to_calc] * phivals)
            phi_sin = np.sin(self.allmodes_l[mode_to_calc] * phivals)

            rgrid, phigrid = np.meshgrid(r, phivals)
            field_r_cos, field_phi = np.meshgrid(phi_cos, field_1d)
            field_r_sin, field_phi = np.meshgrid(phi_sin, field_1d)
            field_cos = field_r_cos * field_phi
            field_sin = field_r_sin * field_phi

            # Normalise each field so its total intensity is 1
            field_cos = field_cos / np.sqrt(np.sum(field_cos**2))
            field_sin = field_sin / np.sqrt(np.sum(field_sin**2))
            field_cos = np.nan_to_num(field_cos)
            field_sin = np.nan_to_num(field_sin)

            field_cos_cart, d = polarTransform.convertToCartesianImage(field_cos.T)
            field_sin_cart, d = polarTransform.convertToCartesianImage(field_sin.T)

            if rotate_mode_angle is not None:
                print('Warning: rotating mode fields by %f degrees. ONLY APPLIES TO CARTESIAN FIELDS!' %
                      rotate_mode_angle)
                field_cos_cart = ndimage.rotate(field_cos_cart, rotate_mode_angle, reshape=False)
                field_sin_cart = ndimage.rotate(field_sin_cart, rotate_mode_angle, reshape=False)

            if normtosum:
                field_cos = field_cos / np.sqrt(np.sum(field_cos**2))
                field_sin = field_sin / np.sqrt(np.sum(field_sin**2))
                field_cos_cart = field_cos_cart / np.sqrt(np.sum(field_cos_cart**2))
                field_sin_cart = field_sin_cart / np.sqrt(np.sum(field_sin_cart**2))

            self.allmodefields_cos_polar.append(field_cos)
            self.allmodefields_cos_cart.append(field_cos_cart)
            self.allmodefields_sin_polar.append(field_sin)
            self.allmodefields_sin_cart.append(field_sin_cart)
            self.allmodefields_rsoftorder.append(field_cos_cart)
            if self.allmodes_l[mode_to_calc] > 0:
                self.allmodefields_rsoftorder.append(field_sin_cart)

            if show_plots:
                self.plot_fiber_modes(mode_to_calc, zlim)
                plt.pause(0.5)


    def plot_fiber_modes(
        self, 
        mode_to_plot, 
        zlim=0.04, 
        ignum=1):
        """
        Make a plot of the cos and sin amplitudes of a given mode

        Parameters
        ----------
        mode_to_plot
            Number of mode to plot
        zlim
            Maximum value to plot
        """
        plt.figure(fignum)
        plt.clf()
        plt.subplot(121)
        sz = self.max_r * self.core_radius
        plt.imshow(self.allmodefields_cos_cart[mode_to_plot], extent=(-sz, sz, -sz, sz), cmap='bwr',
                   vmin=-zlim, vmax=zlim)
        plt.xlabel('Position ($\mu$m)')
        plt.ylabel('Position ($\mu$m)')
        plt.title('Mode l=%d, m=%d (cos)' % (self.allmodes_l[mode_to_plot], self.allmodes_m[mode_to_plot]))
        core_circle = plt.Circle((0,0), self.core_radius, color='k', fill=False, linestyle='--', alpha=0.2)
        plt.gca().add_patch(core_circle)
        plt.subplot(122)
        sz = self.max_r * self.core_radius
        plt.imshow(self.allmodefields_sin_cart[mode_to_plot], extent=(-sz, sz, -sz, sz), cmap='bwr',
                   vmin=-zlim, vmax=zlim)
        plt.xlabel('Position ($\mu$m)')
        plt.title('Mode l=%d, m=%d (sin)' % (self.allmodes_l[mode_to_plot], self.allmodes_m[mode_to_plot]))
        core_circle = plt.Circle((0,0), self.core_radius, color='k', fill=False, linestyle='--', alpha=0.2)
        plt.gca().add_patch(core_circle)
        plt.pause(0.001)
        print('LP mode %d, %d' % (self.allmodes_l[mode_to_plot], self.allmodes_m[mode_to_plot]))


    def plot_injection_field(
        self, 
        field, 
        fignum=1, 
        show_colorbar=True, 
        logI=False, 
        vmin=None):

        sz = self.max_r * self.core_radius
        plt.figure(fignum)
        plt.clf()
        plt.subplot(211)
        if logI:
            im = np.abs(field)**2
            im = np.log10(im / np.max(im))
            plt.imshow(im, extent=(-sz, sz, -sz, sz), vmin=vmin)
        else:
            plt.imshow(np.abs(field), extent=(-sz, sz, -sz, sz))
        core_circle = plt.Circle((0,0), self.core_radius, color='w', fill=False, linestyle='--', alpha=0.2*3)
        plt.gca().add_patch(core_circle)
        plt.xlabel('Position ($\mu$m)')
        plt.ylabel('Position ($\mu$m)')
        if logI:
            plt.title('log Intensity')
        else:
            plt.title('Amplitude')
        if show_colorbar:
            plt.colorbar()
        plt.subplot(212)
        plt.imshow(np.angle(field), extent=(-sz, sz, -sz, sz), cmap='bwr', vmin=-np.pi, vmax=np.pi)
        core_circle = plt.Circle((0,0), self.core_radius, color='w', fill=False, linestyle='--', alpha=0.2*3)
        plt.gca().add_patch(core_circle)
        plt.xlabel('Position ($\mu$m)')
        plt.ylabel('Position ($\mu$m)')
        plt.title('Phase')
        if show_colorbar:
            plt.colorbar()
        plt.tight_layout()


    def make_arb_input_field(
        self, 
        field_type, 
        power=1, 
        location=[0,0], 
        sigma=3, 
        phase=0, 
        add_to_existing=False,
        show_plots=False, 
        logI=False):
        """
        Make an arbitrary input field to inject into MM region
        Parameters
        ----------
        field_type
            Type of field to create. Current options:
                'gaussian'
        power
            Total power in field
        location
            Location (in microns) of field feature, relative to centre
        sigma
            Standard devitaion for Gaussian field
        add_to_existing : bool
            If true, add to current field in self.input_field, rather than replacing it.
        """

        if field_type is 'gaussian':
            posn = np.array(location) / self.microns_per_pixel
            xvals = np.linspace(-self.npix, self.npix, self.npix*2)
            xgrid,ygrid = np.meshgrid(xvals,xvals)
            input_fld_ampl = np.exp(-( (xgrid-posn[0])**2 + (ygrid-posn[1])**2) / (2*sigma**2))
            input_fld_ampl = input_fld_ampl / np.sqrt(np.sum(input_fld_ampl**2)) * np.sqrt(power)
            input_fld_phase = np.zeros((self.npix*2,self.npix*2))
            input_fld_phase = np.ones((self.npix*2,self.npix*2)) * phase
            input_fld = input_fld_ampl * np.exp(1j * input_fld_phase)
        else:
            print('Error: unknown field type specified')
            return

        if add_to_existing:
            if self.input_field is None:
                self.input_field = np.zeros((self.npix*2, self.npix*2), dtype='complex')
            self.input_field = self.input_field + input_fld
        else:
            self.input_field = input_fld

        if show_plots:
            self.plot_injection_field(self.input_field, logI=logI)
            core_circle = plt.Circle((0,0), self.core_radius, color='k', fill=False, linestyle='--', alpha=0.2)
            plt.gca().add_patch(core_circle)


    #
    # def make_turb_input_field(self, power=1, r=8, r0=0.2, inner_scale=1e-3, outer_scale=0.2):
    #     pass
    #


    def calc_injection(
        self, 
        input_field=None, 
        mode_field=None,
        mode_field_number=None, 
        verbose=False):
        """
        Calculate the overlap integral between an input field and fiber mode.
        Parameters
        ----------
        input_field
            Complex input field. If not specified, will use self.input_field
        mode_field
            Complex fiber mode field. EITHER specify this or mode_field_number
        mode_field_number
            Index of fiber mode to use, if mode_field not specified
        verbose : bool
            Print values to console

        Returns
        -------
        overlap_int
            Total coupling efficiency
        """
        if (mode_field is not None) and (mode_field_number is not None):
            print('Error: cannot specify both mode_field and mode_field_number.')
            return
        if input_field is None:
            input_field = self.input_field
        if mode_field_number is not None:
            mode_field = self.make_complex_fld(self.allmodefields_rsoftorder[mode_field_number])
        # overlap_int = np.abs(np.sum(input_field*mode_field))**2 / \
        #               ( np.sum(np.abs(mode_field)**2) * np.sum(np.abs(input_field)**2) )
        overlap_int_complex = np.sum(input_field*mode_field) / \
                      np.sqrt( np.sum(np.abs(mode_field)**2) * np.sum(np.abs(input_field)**2) )
        overlap_int = np.abs(overlap_int_complex)**2

        if verbose:
            print('Total power in input field: %f' % np.sum(np.abs(input_field)**2))
            print('Total power in mode field: %f' % np.sum(np.abs(mode_field)**2))
            print('Injection efficiency: %f' % overlap_int)
            print('Coupled power: %f' % (overlap_int*np.sum(np.abs(input_field)**2)))

        return overlap_int, overlap_int_complex


    def calc_injection_multi(
        self, 
        input_field=None, 
        mode_field_numbers=None, 
        verbose=False, 
        show_plots=False,
        fignum=1, 
        complex=False, 
        logplot=False, 
        modes_to_plot=None, 
        ylim=None,
        return_abspower=False):
        """
        Calculate the overlap integral between an input field and a set of fiber modes.
        Parameters
        ----------
        input_field
            Complex input field. If not specified, will use self.input_field
        mode_field_numbers
            List of indices of fiber modes to use
        verbose : bool
            Print values to console
        show_plots : bool
            Make a plot of coupling per mode
        fignum

        Returns
        -------
        overlap_int
            Total coupling efficiency
        overlap_int_vals
            Array of coupling efficiencies for each mode
        """
        if input_field is None:
            input_field = self.input_field
        overlap_int_vals = []
        overlap_int_vals_complex = []
        for modenum in mode_field_numbers:
            mode_field = self.make_complex_fld(self.allmodefields_rsoftorder[modenum])
            # cur_overlap_int = np.abs(np.sum(input_field*mode_field))**2 / \
            #               ( np.sum(np.abs(mode_field)**2) * np.sum(np.abs(input_field)**2) )
            overlap_int_complex = np.sum(input_field*mode_field) / \
                                  np.sqrt( np.sum(np.abs(mode_field)**2) * np.sum(np.abs(input_field)**2) )
            cur_overlap_int = np.abs(overlap_int_complex)**2
            overlap_int_vals.append(cur_overlap_int)
            overlap_int_vals_complex.append(overlap_int_complex)
            if verbose:
                print('Injection efficiency for mode %d: %f' % (modenum, cur_overlap_int))

        overlap_int_vals = np.array(overlap_int_vals)
        overlap_int_vals_complex = np.array(overlap_int_vals_complex)
        overlap_int = np.sum(overlap_int_vals)
        if verbose:
            print('Total power in input field: %f' % np.sum(np.abs(input_field)**2))
            print('Injection efficiency: %f' % overlap_int)
            print('Coupled power: %f' % (overlap_int*np.sum(np.abs(input_field)**2)))

        if return_abspower:
            inpow = np.sum(np.abs(input_field)**2)
            overlap_int = overlap_int * inpow
            overlap_int_vals = overlap_int_vals * inpow
            overlap_int_vals_complex = overlap_int_vals_complex * np.sqrt(inpow)

        if show_plots:
            plt.figure(fignum)
            plt.clf()
            # plt.plot(mode_field_numbers, overlap_int_vals, '-o')
            if logplot:
                overlap_int_vals_plot = np.log10(overlap_int_vals)
            else:
                overlap_int_vals_plot = overlap_int_vals
            if modes_to_plot is None:
                plt.bar(mode_field_numbers, overlap_int_vals_plot, tick_label=mode_field_numbers)
                plt.xticks(np.arange(0, self.nmodes, 1), rotation=90, fontsize=8)
                plt.gca().set_xticklabels(self.modelabels)
            else:
                plt.bar(mode_field_numbers[modes_to_plot], overlap_int_vals_plot[modes_to_plot],
                    tick_label=mode_field_numbers[modes_to_plot])
            plt.xlabel('Mode number')
            plt.ylabel('Coupling efficiency')
            # plt.title('Total coupling efficiency: %f' % overlap_int)
            plt.title('Total coupling efficiency: %.2g' % overlap_int)
            plt.ylim((0,ylim))
            plt.tight_layout()

        if complex:
            return overlap_int, overlap_int_vals, overlap_int_vals_complex
        else:
            return overlap_int, overlap_int_vals


    def make_complex_fld(
        self, 
        raw_ampl):
        """
        Convert a raw amplitude (where values <0 mean pi phase) to a complex field
        Parameters
        ----------
        raw_ampl
            Input amplitude array
        Returns
        -------
        complex_psf
            The input amplitude rendered as a complex field
        """
        fld_ampl = np.abs(raw_ampl)
        fld_phase = np.zeros_like(raw_ampl)
        fld_phase[raw_ampl < 0] = np.pi
        complex_psf = fld_ampl * np.exp(1j * fld_phase)
        return complex_psf


    def cx2ap(
        self, 
        input):
        ampl = np.abs(input)
        phase = np.angle(input)/np.pi*180
        return ampl, phase


    def saveToRSoft(
        self, 
        complex_psf, 
        outfile="PSFOut", 
        size_data=100, 
        normtomax=False):
        """
        From Theo's FITS2rsoft script.
        - The parameter 'size_data' is the physical half-size of the array in um so make
        sure you have the dimensions correct.

        - The format of the .fld file is 2 columns for each single column in the
        fits files, the real then the imaginary. There is a header at the top of the
        file which shouldn't need changing, it is just how the data is interpreted
        by the program
        """
        complex_psf = np.transpose(complex_psf) # Rsoft seems to use different orientation

        psf_real = complex_psf.real
        psf_imag = complex_psf.imag

        len_data = psf_real.shape[0]

        # Empty array to take all data 3d
        whole_psf = np.zeros([len_data, 2 * len_data])

        # RSoft takes data in columns of real and imaginary parts. (make odd ones imag, even real)
        whole_psf[:, ::2] = psf_real
        whole_psf[:, 1::2] = psf_imag
        if normtomax:
            whole_psf = whole_psf / whole_psf.max()

        # -------------------------- Writing FLD file ----------------------------------
        outfile = outfile + "_inputfield"
        print("Writing field to file " + outfile + ".fld")
        header = (
            "/rn,a,b/nx0\n/rn,qa,qb\n{0} -{1} {1} 0 OUTPUT_REAL_IMAG_3D\n{0} -{1} {1}"
        ).format(len_data, size_data)
        np.savetxt(outfile + ".fld", whole_psf, fmt="%.18E", header=header, comments="")


    def save_multiple_rsoft(
        self, 
        modecoeffs, 
        outpath='./', 
        size_data=100,
        makeBatFile=False, 
        indFile="bptmp.ind",
        outPrefix="BPScan",
        numBatfiles=1, 
        savemetadata=True, 
        show_plots=True, 
        beamprop_prefix='bp_',
        make_hyperbat=False):
        """
        Save multiple rosft launch fields for a range of modes, and .bat file to run them
        Parameters
        ----------
        modecoeffs
            List of coeff vectors, each vector having one coeff per mode
        outpath
        fileprefix
        size_data
        makeBatFile
        indFile
        outPrefix
        numBatfiles
        """
        allOutfiles = []

        imsz = np.shape(self.allmodefields_rsoftorder[0])[0]
        filecount = 0
        for cur_coeffs in modecoeffs:
            print(' ')
            print("Saving .fld file using coeffs:")
            print(cur_coeffs)
            cur_modefield = np.zeros((imsz,imsz), dtype='complex')
            for k in range(self.nmodes):
                cur_modefield = cur_modefield + cur_coeffs[k] * \
                    self.make_complex_fld(self.allmodefields_rsoftorder[k])
            self.input_field = cur_modefield
            outfilename = outPrefix + '_%.2d' % filecount
            self.saveToRSoft(cur_modefield, outfile=outpath+outfilename, size_data=size_data)
            allOutfiles.append(outfilename)
            if show_plots:
                plt.figure(1)
                plt.subplot(121)
                plt.imshow(np.abs(cur_modefield))
                plt.subplot(122)
                plt.imshow(np.angle(cur_modefield))
                plt.pause(0.001)

            if savemetadata:
                metadata_filename = outfilename + '_metadata.npz'
                np.savez(outpath+metadata_filename, outfilename=outfilename, cur_modefield=cur_modefield,
                         cur_coeffs=cur_coeffs, size_data=size_data)
            filecount = filecount + 1

        if makeBatFile:
            allOutfilenames = []
            progname = "bsimw32"

            nf = len(allOutfiles)
            # batfileLength = nf // numBatfiles
            batfileLengths = np.zeros(numBatfiles)
            batfileStarts = np.zeros(numBatfiles)
            batfileEnds = np.zeros(numBatfiles)
            fnum = 0
            for k in range(nf):
                batfileLengths[fnum] = batfileLengths[fnum] + 1
                fnum = fnum + 1
                if fnum == numBatfiles:
                    fnum = 0
            for k in range(numBatfiles):
                if k == 0:
                    batfileStarts[k] = 0
                    batfileEnds[k] = batfileLengths[k]
                else:
                    batfileStarts[k] = batfileEnds[k-1]
                    batfileEnds[k] = batfileStarts[k] + batfileLengths[k]

            allBatfileNames = []

            for k in range(numBatfiles):
                # batfileLength = batfileLengths[k]
                # startInd = int(k * batfileLength)
                # endInd = int((k + 1) * batfileLength)
                startInd = int(batfileStarts[k])
                endInd = int(batfileEnds[k])
                if k == (numBatfiles - 1):
                    curOutfiles = allOutfiles[startInd:]
                else:
                    curOutfiles = allOutfiles[startInd:endInd]
                print("Making .bat file: " + outpath + outPrefix + "_" + str(k) + ".bat")
                batfile = open(outpath + outPrefix + "_" + str(k) + ".bat", "w")
                allBatfileNames.append(outPrefix + "_" + str(k) + ".bat")
                for launch_file in curOutfiles:
                    cmdStr = (progname + " " + indFile + " prefix=" + beamprop_prefix + launch_file + " launch_file="
                            + launch_file + "_inputfield" + ".fld wait=0\n")
                    print(cmdStr)
                    batfile.write(cmdStr)
                    self.all_indiv_commands.append(cmdStr)
                    allOutfilenames.append(beamprop_prefix + launch_file)
                batfile.close()
                self.allBatfileNames.append(allBatfileNames)
            if numBatfiles > 1:
                superbatfile = open(outpath + "runAllBatfiles" + outPrefix + ".bat", "w")
                for batfilename in allBatfileNames:
                    cmdStr = "start cmd /k call " + batfilename + "\n"
                    superbatfile.write(cmdStr)
                superbatfile.close()
                self.all_runallbats.append("runAllBatfiles" + outPrefix + ".bat")

            if make_hyperbat:
                hyperbatfile = open(outpath + "runAllSuperBatfiles" + '_WL%.3f' %  (self.wavelength*1) +
                                    ".bat", "w")
                for batfilename in self.all_runallbats:
                    cmdStr = "start cmd /k call " + batfilename + "\n"
                    hyperbatfile.write(cmdStr)
                hyperbatfile.close()
                self.all_hyperbats.append("runAllSuperBatfiles" + '_WL%.3f' %  (self.wavelength*1) +
                                          ".bat")

    def load_rsoft_data_sm2mm(
        self,
        rsoft_datadir,
        rsoft_fileprefix,
        show_plots=False,
        av_fluxes=100,
        offset_sm_meas=100,
        save_output=False,
        zero_phases=True,
        fignum=2):
        """
        Load rsoft outputs for a single-mode to multi-mode propagation. Assumes one SM waveguide
        excited at a time.

        Parameters
        ----------
        rsoft_datadir
        rsoft_fileprefix
        show_plots : bool
        av_fluxes
            If >0, average n flux measurements from monitor
        offset_sm_meas
            Measure SM fluxes starting at this index. Useful to skip first part of monitor
            since still coupling.
        save_output : bool
            Save the measured powers and phases to a npz file
        zero_phases : bool
            Set the SM input phases to zero
        fignum
            Figure number in which to display plots
        """

        # Specify relevant indices of MONdata:
        sm_power_monrange = (0, 19)
        mm_power_monrange = (19, 38)
        mm_phase_monrange = (38, 57)

        nwgs = self.nmodes # TODO - Fix mixing up of nmodes and nwgs in this function
        for wgnum in range(nwgs):
            rsoft_filename = rsoft_fileprefix + '%.2d' % (wgnum+1)
            print('Reading rsoft files ' + rsoft_filename)
            r = Rsoftdata(rsoft_datadir)
            r.readall(filename=rsoft_filename)
            if show_plots:
                r.plotall()
                plt.pause(0.001)

            smpower_mons = r.MONdata[:, sm_power_monrange[0]:sm_power_monrange[1]]
            mmpower_mons = r.MONdata[:, mm_power_monrange[0]:mm_power_monrange[1]]
            mmphase_mons = r.MONdata[:, mm_phase_monrange[0]:mm_phase_monrange[1]]
            if av_fluxes > 0:
                f = smpower_mons[offset_sm_meas:av_fluxes+offset_sm_meas, :]
                smpower = f.mean(axis=0)
                f = mmpower_mons[-av_fluxes:, :]
                mmpower = f.mean(axis=0)
            else:
                smpower = smpower_mons[offset_sm_meas, :]
                mmpower = mmpower_mons[-1, :]
            mmphase = mmphase_mons[-1, :]

            if zero_phases:
                smphases = np.zeros(nwgs)

            ## Convert to amplitudes
            smpower = np.sqrt(smpower)
            mmpower = np.sqrt(mmpower)

            self.all_smpowers.append(smpower)
            self.all_smphases.append(smphases)
            self.all_mmpowers.append(mmpower)
            self.all_mmphases.append(mmphase)

        if save_output:
            outfilename = self.datadir + 'extractedvals_' + rsoft_fileprefix + '.npz'
            np.savez(outfilename, all_smpowers=self.all_smpowers, all_mmpowers=self.all_mmpowers,
                     all_mmphases=self.all_mmphases, all_smphases=self.all_smphases)

        if show_plots:
            plt.figure(fignum)
            plt.clf()
            plt.subplot(121)
            plt.imshow(np.asarray(self.all_mmpowers))
            plt.colorbar()
            plt.title('Output mode power')
            plt.ylabel('Excited waveguide no.')
            plt.xlabel('Mode no.')
            plt.subplot(122)
            plt.imshow(np.asarray(self.all_mmphases), cmap='twilight_shifted')
            plt.colorbar()
            plt.title('Output mode phase')
            plt.ylabel('Excited waveguide no.')
            plt.xlabel('Mode no.')
            plt.tight_layout()


    def load_rsoft_data_mm2sm(
        self,
        rsoft_datadir,
        rsoft_fileprefix,
        show_plots=False,
        av_fluxes=100,
        save_output=False,
        fignum=2):
        """
        Load rsoft outputs for a multi-mode to single-mode propagation. Assumes one MM mode
        excited at a time.

        Parameters
        ----------
        rsoft_datadir
        rsoft_fileprefix
        show_plots : bool
        av_fluxes
            If >0, average n flux measurements from monitor
        save_output : bool
            Save the measured powers and phases to a npz file
        fignum
            Figure number in which to display plots
        """

        # Specify mode indices
        LP_modes = self.LP_modes

        # Specify relevant indices of MONdata
        sm_power_monrange = (0, 19)
        mm_power_monrange = (19, 38)
        mm_phase_monrange = (38, 57)
        sm_phase_monrange = (57, 76)

        nwgs = self.nmodes # TODO - Fix mixing up of nmodes and nwgs in this function
        for wgnum in range(nwgs):
            rsoft_suffix = 'LP%d%d' % (LP_modes[wgnum,0], LP_modes[wgnum,1])
            rsoft_filename = rsoft_fileprefix + rsoft_suffix
            print('Reading rsoft files ' + rsoft_filename)
            r = Rsoftdata(rsoft_datadir)
            r.readall(filename=rsoft_filename)
            if show_plots:
                r.plotall()
                plt.pause(0.001)

            smpower_mons = r.MONdata[:, sm_power_monrange[0]:sm_power_monrange[1]]
            mmpower_mons = r.MONdata[:, mm_power_monrange[0]:mm_power_monrange[1]]
            mmphase_mons = r.MONdata[:, mm_phase_monrange[0]:mm_phase_monrange[1]]
            smphase_mons = r.MONdata[:, sm_phase_monrange[0]:sm_phase_monrange[1]]

            if av_fluxes > 0:
                f = smpower_mons[-av_fluxes:, :]
                smpower = f.mean(axis=0)
                f = mmpower_mons[0:av_fluxes, :]
                mmpower = f.mean(axis=0)
            else:
                smpower = smpower_mons[-1, :]
                mmpower = mmpower_mons[0, :]
            mmphase = mmphase_mons[0, :]
            smphase = smphase_mons[-1, :]

            ## Convert to amplitudes
            smpower = np.sqrt(smpower)
            mmpower = np.sqrt(mmpower)

            self.all_smpowers.append(smpower)
            self.all_mmpowers.append(mmpower)
            self.all_mmphases.append(mmphase)
            self.all_smphases.append(smphase)

        if save_output:
            outfilename = self.datadir + 'extractedvals_' + rsoft_fileprefix + '.npz'
            np.savez(outfilename, all_smpowers=self.all_smpowers, all_mmpowers=self.all_mmpowers,
                     all_mmphases=self.all_mmphases, all_smphases=self.all_smphases)

        if show_plots:
            plt.figure(fignum)
            plt.clf()
            plt.subplot(121)
            plt.imshow(np.asarray(self.all_smpowers))
            plt.colorbar()
            plt.title('Output waveguide power')
            plt.ylabel('Excited mode no.')
            plt.xlabel('Waveguide no.')
            plt.subplot(122)
            plt.imshow(np.asarray(self.all_smphases), cmap='twilight_shifted')
            plt.colorbar()
            plt.title('Output waveguide phase')
            plt.ylabel('Excited mode no.')
            plt.xlabel('Waveguide no.')
            plt.tight_layout()


    def load_rsoft_data_customfld(
        self,
        rsoft_datadir,
        rsoft_fileprefix,
        indfile, 
        show_plots=False,
        av_fluxes=100,
        save_output=False,
        fignum=2,
        LP_file_numbering=False,
        zero_mmphase=True,
        np_fileprefix=None,
        ap_rad=30,
        show_indivmasks=False,
        use_pathway_mons=False,
        use_monitor_objects=False,
        reorder_monobjs=False,
        nfiles=None,
        save_label=''):
        """
        Load rsoft outputs for MM to SM simulations, where the input FLD was generated from this code
        and no monitors are used.

        Parameters
        ----------
        rsoft_datadir
        rsoft_fileprefix
        show_plots : bool
        av_fluxes
            If >0, average n flux measurements from monitor
        save_output : bool
            Save the measured powers and phases to a npz file
        fignum
            Figure number in which to display plots
        """

        sm_power_monrange = (0, 19)
        sm_phase_monrange = (19, 38)

        # Specify mode indices
        LP_modes = self.LP_modes

        if nfiles is None:
            nfiles = self.nmodes
        self.all_smpowers = []
        self.all_mmpowers = []
        self.all_mmphases = []
        self.all_smphases = []
        self.out_field_ampl = []
        self.out_field_phase = []
        if np_fileprefix is None:
            np_fileprefix = rsoft_fileprefix
        if use_pathway_mons and use_monitor_objects:
            print("Error - can't simultaneously specify both tyes of monitors")
            return
        self.all_FLDampls = []
        for modenum in range(nfiles):

            # Get SM WG output amps and phases from rsoft output files
            if LP_file_numbering:
                rsoft_suffix = 'LP%d%d' % (LP_modes[modenum,0], LP_modes[modenum,1])
            else:
                rsoft_suffix = '%.2d' % modenum
            rsoft_filename = rsoft_fileprefix + rsoft_suffix

            print('Reading rsoft files ' + rsoft_filename)
            r = Rsoftdata(rsoft_datadir)

            if use_pathway_mons:
                r.loadMON(filename=rsoft_filename)
                smpower_mons = r.MONdata[:, sm_power_monrange[0]:sm_power_monrange[1]]
                smphase_mons = r.MONdata[:, sm_phase_monrange[0]:sm_phase_monrange[1]]
                if av_fluxes > 0:
                    f = smpower_mons[-av_fluxes:, :]
                    smpower = f.mean(axis=0)
                else:
                    smpower = smpower_mons[-1, :]
                smphase = smphase_mons[-1, :]
                smpower = np.sqrt(smpower)
                self.all_smpowers.append(smpower)
                self.all_smphases.append(smphase)
            elif use_monitor_objects:
                print('Loading monitor filename: '+rsoft_filename)
                try:
                    r.loadMONOBJ(filename=rsoft_filename)
                    ampls = r.output_ampls
                    phases = r.output_phases
                    if reorder_monobjs:
                        inds = np.array(self.monitor_order) - 1
                        ampls = ampls[inds]
                        phases = phases[inds]
                except:
                    print('ERROR: could not open file '+rsoft_filename)
                    # print('Replacing with zeros')
                    # ampls = np.zeros(self.nwgs)
                    # phases = np.zeros(self.nwgs)
                self.all_smpowers.append(ampls)
                self.all_smphases.append(phases)
            else:
                r.loadFLD(filename=rsoft_filename)
                self.out_field_ampl.append(r.FLDampl)
                self.out_field_phase.append(r.FLDphase)
                self.read_ind_file(rsoft_datadir, indfile, skipfirst=True, getWGposns=True)
                self.measure_wg_fields(show_plots=True, field_index=modenum, show_indivmasks=show_indivmasks,
                                       ap_rad=ap_rad)
                self.all_FLDampls.append(r.FLDampl)
                if show_plots:
                    plt.imshow(r.FLDampl)
                    plt.pause(0.001)
                    # input("Press Enter to continue...")

            # Get input mode powers and phase from npz files
            npfilename = rsoft_datadir+np_fileprefix+rsoft_suffix
            npfile = np.load(npfilename+'_metadata.npz')
            coeffs = npfile['cur_coeffs']
            coeffs = coeffs[:self.nmodes] # Remove coeffs for modes not being used

            mm_power = np.abs(coeffs)
            mm_phase = np.angle(coeffs)/np.pi*180
            self.all_mmpowers.append(mm_power)
            self.all_mmphases.append(mm_phase)
            # if zero_mmphase:
            #     self.all_mmphases.append(np.zeros_like(coeffs))
            # else:
            #     pass

        if save_output:
            # outfilename = self.datadir + 'extractedvals_' + rsoft_fileprefix + '.npz'
            if len(save_label) > 0:
                save_label = save_label + '_'
            outfilename = self.datadir + 'extractedvals_' + save_label + rsoft_fileprefix[3:-1] + '.npz'
            np.savez(outfilename, all_smpowers=self.all_smpowers, all_mmpowers=self.all_mmpowers,
                     all_mmphases=self.all_mmphases, all_smphases=self.all_smphases)

        if show_plots:
            plt.figure(fignum)
            plt.clf()
            plt.subplot(121)
            plt.imshow(np.asarray(self.all_smpowers))
            plt.colorbar()
            plt.title('Output waveguide power')
            plt.ylabel('Excited mode no.')
            plt.xlabel('Waveguide no.')
            plt.subplot(122)
            plt.imshow(np.asarray(self.all_smphases), cmap='twilight_shifted')
            plt.colorbar()
            plt.title('Output waveguide phase')
            plt.ylabel('Excited mode no.')
            plt.xlabel('Waveguide no.')
            plt.tight_layout()


    def load_savedvalues(
        self,
        filename):
        """
        Load previously saved powers and phases from npz file

        Parameters
        ----------
        filename
        """

        f = np.load(self.datadir+filename)
        self.all_smpowers = f['all_smpowers']
        self.all_mmpowers = f['all_mmpowers']
        self.all_mmphases = f['all_mmphases']
        try:
            self.all_smphases = f['all_smphases']
        except:
            self.all_smphases = None


    def set_mmvals_nominal(
        self, 
        quare=False):
        """
        Set all appropriate mm powers and phases to 1 and 0 respectively.
        """

        if square:
            self.all_mmpowers = np.diag(np.ones(self.nmodes))
            self.all_mmphases = np.zeros((self.nmodes, self.nmodes))
        else:
            self.all_mmpowers = np.zeros(self.nmodes)
            self.all_mmpowers[0] = 1
            self.all_mmphases = np.zeros(self.nmodes)


    def set_smvals_nominal(
        self, 
        square=False):
        """
        Set all appropriate sm powers and phases to 1 and 0 respectively.
        """

        if square:
            self.all_smpowers = np.diag(np.ones(self.nmodes))
            self.all_smphases = np.zeros((self.nmodes, self.nmodes))
        else:
            self.all_smpowers = np.zeros(self.nmodes)
            self.all_smpowers[0] = 1
            self.all_smphases = np.zeros(self.nmodes)


    def make_transfer_matrix_sm2mm(
        self, 
        sm_phase=0):
        """
        Calculate the transfer matrix Cmat from SM-to-MM simulations.
        This requires that for each measurement, only one waveguide is excited

        Parameters
        ----------
        sm_phase
            Phase to set the SM waveguide excitation to
        """

        nwgs = len(self.all_smpowers) # TODO - fix mixing-up of nmodes and nwgs here
        Cmat = np.zeros((nwgs,nwgs), dtype=np.cfloat)
        for col_num in range(nwgs):
            sm_val = self.all_smpowers[col_num][col_num] * np.exp(1j*sm_phase)
            col = self.all_mmpowers[col_num] * np.exp(1j*self.all_mmphases[col_num]/180*np.pi) / sm_val
            Cmat[:,col_num] = col
        self.Cmat = Cmat


    def make_transfer_matrix_mm2sm(
        self, 
        mm_phase=None, 
        show_plots=True, 
        truncate=None):
        """
        Calculate the transfer matrix Dmat from MM-to-SM simulations.
        This requires that for each measurement, only one mode is excited

        Parameters
        ----------
        mm_phase
            Phase to set MM mode phase to. If None, use phase from data.
        """

        # nwgs = len(self.all_mmpowers)
        Cmat = np.zeros((self.nwgs,self.nmodes), dtype=np.cfloat)
        for col_num in range(self.nmodes):
            if mm_phase is not None:
                mm_val = self.all_mmpowers[col_num][col_num] * np.exp(1j*mm_phase)
            else:
                mm_val = self.all_mmpowers[col_num][col_num] * np.exp(1j*self.all_mmphases[col_num][col_num])
            col = self.all_smpowers[col_num] * np.exp(1j*self.all_smphases[col_num]/180*np.pi) / mm_val
            Cmat[:,col_num] = col
        # if truncate is not None:
        #     Cmat = Cmat[:truncate+1, :truncate+1]
        #     self.nwgs = truncate
        #     self.nmodes = truncate
        self.Cmat = Cmat
        if show_plots:
            self.plot_matrix()

        # Make a matrix mapping to the output intensities
        # real_mat = np.vstack()


    def plot_matrix(
        self, 
        matrix=None, 
        fignum=3, 
        cmap='twilight_shifted', 
        figsize=(9,4)):

        if matrix is None:
            matrix = self.Cmat
        plt.figure(fignum, figsize=figsize)
        plt.clf()
        plt.subplot(121)
        plt.imshow(np.abs(matrix))
        plt.colorbar()
        plt.title('Matrix amplitude')
        plt.xlabel('Excited mode no.')
        plt.ylabel('Waveguide no.')
        plt.subplot(122)
        plt.imshow(np.angle(matrix), cmap=cmap)
        plt.colorbar()
        plt.title('Matrix phase')
        plt.xlabel('Excited mode no.')
        plt.ylabel('Waveguide no.')
        # plt.tight_layout()


    def matrix_complex2real(
        self,
        Cmat):
        # TODO - Do this without a loop (Kronecker product?)
        Rmat = np.zeros((Cmat.shape[0]*2, Cmat.shape[1]*2))
        for m in range(Cmat.shape[0]):
            for n in range(Cmat.shape[1]):
                a = np.real(Cmat[m,n])
                b = np.imag(Cmat[m,n])
                block = np.array([[a, -b], [b, a]])
                Rmat[m*2:m*2+2, n*2:n*2+2] = block
        return Rmat


    def matrix_real2complex(
        self, 
        Rmat):
        # TODO - Do this without a loop (Kronecker product?)
        Cmat = np.zeros((Rmat.shape[0]//2, Rmat.shape[1]//2), dtype='complex')
        for m in range(Cmat.shape[0]):
            for n in range(Cmat.shape[1]):
                block = Rmat[m*2:m*2+2, n*2:n*2+2]
                a = block[0,0]
                b = block[1,0]
                Cmat[m,n] = a + b*1j
        return Cmat


    def load_rsoft_data_sm2mm_single(
        self, 
        rsoft_datadir, 
        rsoft_fileprefix, 
        show_plots=False,
        av_fluxes=100, 
        offset_sm_meas=100, 
        zero_phases=True):
        """
        Load rsoft outputs for a single propagation (sm2mm).

        Parameters
        ----------
        rsoft_datadir
        rsoft_fileprefix
        show_plots : bool
        av_fluxes
            If >0, average n flux measurements from monitor
        offset_sm_meas
            Measure SM fluxes starting at this index. Useful to skip first part of monitor
            since still coupling.
        """

        self.all_smpowers = []
        self.all_mmpowers = []
        self.all_mmphases = []
        self.all_smphases = []

        # Specify relevant indices of MONdata:
        sm_power_monrange = (0, 19)
        mm_power_monrange = (19, 38)
        mm_phase_monrange = (38, 57)

        rsoft_filename = rsoft_fileprefix
        print('Reading rsoft files ' + rsoft_filename)
        r = Rsoftdata(rsoft_datadir)
        r.readall(filename=rsoft_filename)
        if show_plots:
            r.plotall()
            plt.pause(0.001)

        self.MONdata = r.MONdata ## TESTING

        smpower_mons = r.MONdata[:, sm_power_monrange[0]:sm_power_monrange[1]]
        mmpower_mons = r.MONdata[:, mm_power_monrange[0]:mm_power_monrange[1]]
        mmphase_mons = r.MONdata[:, mm_phase_monrange[0]:mm_phase_monrange[1]]
        if av_fluxes > 0:
            f = smpower_mons[offset_sm_meas:av_fluxes+offset_sm_meas, :]
            smpower = f.mean(axis=0)
            f = mmpower_mons[-av_fluxes:, :]
            mmpower = f.mean(axis=0)
        else:
            smpower = smpower_mons[offset_sm_meas, :]
            mmpower = mmpower_mons[-1, :]
        mmphase = mmphase_mons[-1, :]

        if zero_phases:
            smphases = np.zeros(self.nmodes) # TODO - Fix mixing up of nmodes and nwgs in this function

        ## Convert to amplitudes
        smpower = np.sqrt(smpower)
        mmpower = np.sqrt(mmpower)

        self.all_smpowers.append(smpower)
        self.all_smphases.append(smphases)
        self.all_mmpowers.append(mmpower)
        self.all_mmphases.append(mmphase)


    def load_rsoft_data_mm2sm_single(
        self, 
        rsoft_datadir, 
        rsoft_fileprefix, 
        show_plots=False,
        av_fluxes=100):
        """
        Load rsoft outputs for a single propagation (mm2sm).

        Parameters
        ----------
        rsoft_datadir
        rsoft_fileprefix
        show_plots : bool
        av_fluxes
            If >0, average n flux measurements from monitor
        """

        # Specify mode indices
        LP_modes = self.LP_modes

        self.all_smpowers = []
        self.all_mmpowers = []
        self.all_mmphases = []
        self.all_smphases = []

        # Specify relevant indices of MONdata
        sm_power_monrange = (0, 19)
        mm_power_monrange = (19, 38)
        mm_phase_monrange = (38, 57)
        sm_phase_monrange = (57, 76)

        rsoft_filename = rsoft_fileprefix
        print('Reading rsoft files ' + rsoft_filename)
        r = Rsoftdata(rsoft_datadir)
        r.readall(filename=rsoft_filename)
        if show_plots:
            r.plotall()
            plt.pause(0.001)

        smpower_mons = r.MONdata[:, sm_power_monrange[0]:sm_power_monrange[1]]
        mmpower_mons = r.MONdata[:, mm_power_monrange[0]:mm_power_monrange[1]]
        mmphase_mons = r.MONdata[:, mm_phase_monrange[0]:mm_phase_monrange[1]]
        smphase_mons = r.MONdata[:, sm_phase_monrange[0]:sm_phase_monrange[1]]

        if av_fluxes > 0:
            f = smpower_mons[-av_fluxes:, :]
            smpower = f.mean(axis=0)
            f = mmpower_mons[0:av_fluxes, :]
            mmpower = f.mean(axis=0)
        else:
            smpower = smpower_mons[-1, :]
            mmpower = mmpower_mons[0, :]
        mmphase = mmphase_mons[0, :]
        smphase = smphase_mons[-1, :]

        ## Convert to amplitudes
        smpower = np.sqrt(smpower)
        mmpower = np.sqrt(mmpower)

        self.all_smpowers.append(smpower)
        self.all_mmpowers.append(mmpower)
        self.all_mmphases.append(mmphase)
        self.all_smphases.append(smphase)

        self.out_field_ampl = []
        self.out_field_phase = []
        self.out_field_ampl.append(r.FLDampl)
        self.out_field_phase.append(r.FLDphase)


    def load_rsoft_data_fldonly(
        self, 
        rsoft_datadir, 
        rsoft_fileprefix):

        rsoft_filename = rsoft_fileprefix
        print('Reading rsoft files ' + rsoft_filename)
        r = Rsoftdata(rsoft_datadir)
        r.loadFLD(filename=rsoft_filename)
        self.out_field_ampl.append(r.FLDampl)
        self.out_field_phase.append(r.FLDphase)
        self.all_smpowers = [0]
        self.all_mmpowers = [0]
        self.all_mmphases = [0]
        self.all_smphases = [0]


    def show_outfield(
        self, 
        fignum=1):
        """
        Show teh maps of output amplitude and phase
        Parameters
        ----------
        fignum
            Figure number for plot
        """
        plt.figure(fignum)
        plt.clf()
        plt.subplot(121)
        plt.imshow(self.out_field_ampl[0])
        plt.title('Amplitude')
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(self.out_field_phase[0], cmap='twilight_shifted')
        plt.title('Phase')
        plt.colorbar()
        plt.tight_layout()


    def test_matrix(
        self, 
        matrix, 
        input_powers, 
        input_phases, 
        output_powers, 
        output_phases, 
        fignum=1, 
        pausetime=1,
        num_to_show=None, 
        unnormalise_smpower=True, 
        unwrap_phase=True):
        """
        Plot the outputs from a set of inputs using the transfer matrix, and overplot
        the 'true' values measured.

        Parameters
        ----------
        matrix
            Matrix to use for testing
        input_powers
            List or array of input powers (one row / item per measurement)
        input_phases
            List or array of input phases (one row / item per measurement)
        output_powers
            List or array of output powers (one row / item per measurement)
        output_phases
            List or array of output phases (one row / item per measurement)
        fignum
            Figure number in which to display plots
        pausetime
            Time (s) to wait between plots
        num_to_show
            Which entry of the input list to plot
        unnormalise_smpower : bool
            If rsoft is normalising monitor power by input power, this will 'un-normalise'
            the monitor power by multiplying it by the total input power.
        unwrap_phase
            Add 360 degrees to phases < 0
        """

        plt.figure(fignum)
        nmeas = len(input_powers)
        for n in range(nmeas):
            if unnormalise_smpower:
                output_power = output_powers[n] * np.sqrt(np.sum(input_powers[n]**2))
            else:
                output_power = output_powers[n]
            plt.figure(fignum)
            plt.clf()
            plt.subplot(211)
            plt.plot(output_power,'-x')
            input_val = input_powers[n] * np.exp(1j*input_phases[n]/180*np.pi)
            self.pred_pow = np.abs(np.matmul(matrix, input_val))
            plt.plot(self.pred_pow,'-+')
            plt.ylabel('SM output amplitudes')
            plt.subplot(212)
            plt.plot(output_phases[n],'-x')
            newangs = np.angle(np.matmul(matrix, input_val))/np.pi*180
            if unwrap_phase:
                newangs[newangs < 0] = newangs[newangs < 0]+360
            self.pred_phase = newangs
            plt.plot(newangs,'-+')
            plt.ylabel('SM output phase')
            plt.xlabel('Waveguide number)')
            plt.pause(0.001)

            plt.figure(fignum+1)
            plt.clf()
            plt.subplot(211)
            plt.plot(input_powers[n], 'x-')
            plt.title('Input mode power')
            plt.subplot(212)
            plt.plot(input_phases[n], 'x-')
            plt.title('Input mode phase')
            plt.tight_layout()
            plt.pause(pausetime)

            if num_to_show is not None:
                if num_to_show == n:
                    break


    # def test_matrix_single(self, matrix, input_powers, input_phases, output_powers, output_phases, fignum=1,
    #                        pausetime=1):
    #     """
    #     Plot the outputs from a set of inputs using the transfer matrix, and overplot
    #     the 'true' values measured.
    #
    #     Parameters
    #     ----------
    #     matrix
    #         Matrix to use for testing
    #     input_powers
    #         List or array of input powers (one row / item per measurement)
    #     input_phases
    #         List or array of input phases (one row / item per measurement)
    #     output_powers
    #         List or array of output powers (one row / item per measurement)
    #     output_phases
    #         List or array of output phases (one row / item per measurement)
    #     fignum
    #         Figure number in which to display plots
    #     pausetime
    #         Time (s) to wait between plots
    #     """
    #     plt.figure(fignum)
    #     plt.clf()
    #     plt.subplot(211)
    #     plt.plot(output_powers,'-x')
    #     input_val = input_powers * np.exp(1j*input_phases/180*np.pi)
    #     # plt.plot(np.abs(np.matmul(matrix, input_powers[n])),'-+')
    #     plt.plot(np.abs(np.matmul(matrix, input_val)),'-+')
    #     plt.ylabel('SM output amplitudes')
    #     plt.subplot(212)
    #     plt.plot(output_phases,'-x')
    #     # newangs = np.angle(np.matmul(matrix, input_powers[n]))/np.pi*180
    #     newangs = np.angle(np.matmul(matrix, input_val))/np.pi*180
    #     newangs[newangs < 0] = newangs[newangs < 0]+360
    #     plt.plot(newangs,'-+')
    #     plt.ylabel('SM output phase')
    #     plt.xlabel('Waveguide number)')
    #     plt.pause(pausetime)


    def read_ind_file(
        self, 
        rsoft_datadir, 
        ind_filename, 
        skipfirst=True, 
        getWGposns=False):
        """
        Read useful info from an rsoft .ind file, such as the power and phase of launch
        fields.

        Parameters
        ----------
        rsoft_datadir
        ind_filename
        skipfirst
            Choose whether to skip the first occurrences of power and phase, since they are repeated.
        getWGposns
            If True, return a list of output waveguide positions
        Returns
        -------
        all_powers
        all_phases
        """

        with open(rsoft_datadir+ind_filename, 'r') as indfile:
            indfile_data = indfile.readlines()

        all_powers = []
        all_phases = []
        if skipfirst:
            skip_this_power = True # Skip the first occurrences, since they are repeated.
            skip_this_phase = True
        else:
            skip_this_power = False
            skip_this_phase = False
        for line in indfile_data:
            if 'launch_power' in line:
                if skip_this_power:
                    skip_this_power = False
                else:
                    ll = line.split('=')
                    power = np.float(ll[1].strip())
                    # Convert to amplitude:
                    power = np.sqrt(power)
                    all_powers.append(power)
            if 'launch_phase' in line:
                if skip_this_phase:
                    skip_this_phase = False
                else:
                    ll = line.split('=')
                    phase = np.float(ll[1].strip())
                    all_phases.append(phase)

        if getWGposns:
            all_xposns = []
            all_yposns = []
            # These are params from the Rsoft Symbols
            A = 60
            T = 10
            s = 3/2 * (A/np.sqrt(3))
            gridsize = 0.25
            sq3 = np.sqrt(3)
            r = 15
            print('WARNING! Getting WGposns using fixed value of r=15 um')
            # Pathways aren't necessarily in same order as segments in rsoft, so put order here
            path2segs = [10, 9, 14, 15, 11, 6, 5, 4, 3, 8, 13, 18, 19, 16, 17, 12, 7, 2, 1]
            for line in indfile_data:
                if 'begin.x' in line:
                    ll = line.split('=')
                    expr = ll[1].strip()
                    xval = eval(expr)  / gridsize
                    all_xposns.append(xval)
                if 'begin.y' in line:
                    ll = line.split('=')
                    expr = ll[1].strip()
                    yval = eval(expr)  / gridsize
                    all_yposns.append(yval)
                if len(all_yposns) == self.nwgs: # Was self.nmodes
                    break

            all_final_wgposns = np.zeros((2, self.nwgs)) # Was self.nmodes
            for k in range(self.nwgs): # Was self.nmodes
                all_final_wgposns[1,k] = all_xposns[path2segs[k]-1]
                all_final_wgposns[0,k] = all_yposns[path2segs[k]-1]
            all_final_wgposns = all_final_wgposns + len(self.out_field_ampl[0])/2-1
            self.wg_posns = all_final_wgposns
            return all_powers, all_phases, all_final_wgposns

        return all_powers, all_phases


    def measure_wg_fields(
        self, 
        ap_rad = 30, 
        show_plots=True, 
        show_indivmasks=False, 
        fignum=1, 
        field_index=0):

        ampl_im = self.out_field_ampl[field_index]
        phase_im = self.out_field_phase[field_index]
        if show_plots:
            plt.figure(fignum)
            plt.clf()
            plt.imshow(ampl_im)

        imsz = ampl_im.shape[0]
        Y, X = np.ogrid[:imsz, :imsz]
        ampls = []
        phases = []
        for k in range(self.nwgs):
            dist = np.sqrt((X-self.wg_posns[0,k])**2 + (Y-self.wg_posns[1,k])**2)
            mask = dist <= ap_rad
            maskedim = np.ma.array(ampl_im, mask=~mask)
            if show_indivmasks:
                plt.clf()
                plt.imshow(maskedim)
                plt.pause(0.1)
            ampls.append(np.sum(maskedim))
            maskedim = np.ma.array(phase_im, mask=~mask)
            if show_indivmasks:
                plt.clf()
                plt.imshow(maskedim)
                plt.pause(0.001)
            phases.append(np.mean(maskedim))

        ampls = np.array(ampls)
        phases = np.array(phases)

        # self.all_smpowers = []
        # self.all_smphases = []
        self.all_smpowers.append(ampls)
        self.all_smphases.append(phases)


    def normalise_smpowers(self):
        for k in range(len(self.all_smpowers)):
            self.all_smpowers[k] = self.all_smpowers[k] / np.sqrt(np.sum(np.array(self.all_smpowers[k])**2))


    def make_rsoft_launch_fields(
        self, 
        set_type, 
        num_outs=1, 
        npix=200, 
        max_r=2, 
        indfile=None, 
        show_plots=False,
        make_bat_file=False, 
        num_bat_files=1,
        outpath=None, 
        outprefix=None,
        make_hyperbat=False):
        """
        Create a set of launch fields for an rsoft beamprop sim, and create batch files to run a big analysis
        Parameters
        ----------
        set_type : str
            Which type of set of fields to make. Options are:
                'probe' - Make set of fields exciting one input mode at a time, each with power=1 and phase=0
                'randampphase' - Make a set of fields each with randomly chosen amplitude and phase.
        num_outs
            How many output fields to make - currently only for 'randampphase'
        npix
            Half-width of mode field calculation in pixels
        max_r
            Maximum radius to calculate mode field, where r=1 is the core diameter
        indfile
            Name of rsoft .ind file for beamprop batch files
        show_plots : bool
            Show plots
        make_bat_file : bool
            Save .bat files to run a big rsoft analysis
        num_bat_files
            Number of batch files (for parallel processing)
        outpath
            .fld and batch file output path
        outprefix
            File prefix for output filenames
        """

        self.max_r = max_r
        array_size_microns = self.max_r * self.core_radius * 2
        microns_per_pixel = array_size_microns / (npix*2)

        if self.allmodefields_rsoftorder is None:
            print('No existing fiber modes found, making modes using defaults.')
            self.find_fiber_modes()
            self.make_fiber_modes(show_plots=show_plots, npix=npix)

        if set_type is 'probe':
            modecoeffs = []
            for k in range(self.nmodes):
                coeff = np.zeros(self.nmodes)
                coeff[k] = 1
                modecoeffs.append(coeff)
            self.save_multiple_rsoft(modecoeffs, outpath=outpath, outPrefix=outprefix, size_data=array_size_microns/2,
                                  indFile=indfile, makeBatFile=make_bat_file, numBatfiles=num_bat_files,
                                    make_hyperbat=make_hyperbat)

        elif set_type is 'randampphase':
            loval = 0
            hival = 1
            loval_phase = 0
            hival_phase = 2*np.pi
            modecoeffs = []
            for k in range(num_outs):
                mode_amps = np.random.rand(self.nmodes) * (hival-loval) + loval
                mode_phases = np.random.rand(self.nmodes) * (hival_phase-loval_phase) + loval_phase
                modecoeff = mode_amps * np.exp(1j*mode_phases)
                modecoeffs.append(modecoeff)
            self.save_multiple_rsoft(modecoeffs, outpath=outpath, outPrefix=outprefix, size_data=array_size_microns/2,
                                     indFile=indfile, makeBatFile=make_bat_file, numBatfiles=num_bat_files,
                                     make_hyperbat=make_hyperbat)


    def make_monmodes(
        self, 
        indfile, 
        data_dir='./',
        prefix=None, 
        all_wls=None):

        if all_wls is None:
            all_wls = self.all_wls
        else:
            self.all_wls = all_wls
        n_wls = len(all_wls)
        # source_string = 'free_space_wavelength = %.3f' % orig_wl
        # for k in range(n_wls):
        #     with open(data_dir+indfile, 'r') as infile:
        #         new_string = 'free_space_wavelength = %.3f' % all_wls[k]
        #         infiledata = infile.read()
        #         infiledata = infiledata.replace(source_string, new_string)
        #     outfilename = Path(indfile).stem + '_WL%.3f.ind' %  all_wls[k]*1000
        #     print('Outfile: ' + outfilename)
        #     with open(data_dir+outfilename, 'w') as outfile:
        #         outfile.write(infiledata)
        indfilename = Path(indfile).stem
        if prefix is None:
            prefix = indfilename
        batfile = open(data_dir + prefix + "_RunMultiWlFemsim.bat", "w")
        for k in range(n_wls):
            cmdStr = ('femsim ' + indfilename + ' prefix=' + prefix + '_WL%.3f' %  (all_wls[k]*1) +
                      ' free_space_wavelength=%.3f' % all_wls[k] + ' wait=0\n')
            # cmdStr = ('start cmd /k call femsim ' + indfilename + ' prefix=' + prefix + '_WL%.3f' %  (all_wls[k]*1) +
            #           ' free_space_wavelength=%.3f' % all_wls[k] + ' wait=0\ntimeout 1\n')
            batfile.write(cmdStr)
        batfile.close()
        self.all_hyperbats.append(prefix + "_RunMultiWlFemsim.bat")


    def unpack_cvec(
        self,
        input_vec, 
        amp_phase=False):

        n_out = len(input_vec)*2
        out_vec = np.zeros(n_out)
        for k in range(n_out):
            inmode_num = np.int(k/2)
            if k%2 == 0:
                if amp_phase:
                    out_vec[k] = np.abs(input_vec[inmode_num])
                else:
                    out_vec[k] = np.real(input_vec[inmode_num])
            else:
                if amp_phase:
                    out_vec[k] = np.angle(input_vec[inmode_num])
                else:
                    out_vec[k] = np.imag(input_vec[inmode_num])
        return out_vec


    def make_sim_data(
        self, 
        ndata=1, 
        in_amp_phase=None, 
        amp_range=[0,1], 
        phase_range=[0, 2*np.pi],
        limit_modes=None):

        input_modecoeffs = []
        output_smvals = []
        loval, hival = amp_range
        loval_phase, hival_phase = phase_range
        if in_amp_phase is not None: # Just make a single sample as specified
            input_modecoeffs = in_amp_phase[0] * np.exp(1j*in_amp_phase[1])
            output_smvals = self.Cmat @ input_modecoeffs
        else:
            if limit_modes is None:
                makenmodes = self.nmodes
                Cmat = self.Cmat
            else:
                makenmodes = limit_modes
                Cmat = self.Cmat[:,:limit_modes]
            for k in range(ndata): # Make a random set
                mode_amps = np.random.rand(makenmodes) * (hival-loval) + loval
                mode_phases = np.random.rand(makenmodes) * (hival_phase-loval_phase) + loval_phase
                modecoeff = mode_amps * np.exp(1j*mode_phases)
                input_modecoeffs.append(modecoeff)
                output_vals = Cmat @ modecoeff
                output_smvals.append(output_vals)
                if k % 10000 == 0:
                    print('Making sample %d' % k)
        return input_modecoeffs, output_smvals


    def generate_sim_I_data(
        self, 
        ndata=1, 
        in_amp_phase=None, 
        amp_range=[0,1], 
        phase_range=[0, 2*np.pi],
        outfilename=None, 
        return_output=False, 
        amp_phase=False, 
        limit_modes=None):

        simdata_inputs, simdata_outputs = self.make_sim_data(ndata, amp_range=amp_range,
                                                 limit_modes=limit_modes, phase_range=phase_range)

        # Unpack input data into real,imag pairs
        simdata_input_upk = []
        for invec in simdata_inputs:
            invec_upk = self.unpack_cvec(invec, amp_phase=amp_phase)
            simdata_input_upk.append(invec_upk)

        # Make output data into intensities
        simdata_outputI = []
        for outvec in simdata_outputs:
            # outvec_upk = unpack_cvec(outvec)
            simdata_outputI.append(np.abs(outvec)**2)

        # Save to file as numpy arrays
        simdata_input_arr = np.array(simdata_input_upk)
        simdata_outputI_arr = np.array(simdata_outputI)

        if outfilename is not None:
            np.savez(self.datadir+outfilename, simdata_input_arr=simdata_input_arr,
                     simdata_outputI_arr=simdata_outputI_arr)
            print('Saved %d simulated data examples to file ' % ndata + outfilename)

        if return_output:
            return simdata_input_arr, simdata_outputI_arr

        return


    # def make_sim_data_inverse(self, ndata=1, amp_range=[0,1], phase_range=[0, 2*np.pi],  use_transpose=False):
    #     output_modecoeffs = []
    #     input_smvals = []
    #     loval, hival = amp_range
    #     loval_phase, hival_phase = phase_range
    #
    #     if use_transpose:
    #         mat = np.matrix(self.Cmat).H
    #     else:
    #         mat = np.matrix(self.Cmat).I
    #
    #     for k in range(ndata): # Make a random set
    #         wg_amps = np.random.rand(self.nwgs) * (hival-loval) + loval
    #         wg_phases = np.random.rand(self.nwgs) * (hival_phase-loval_phase) + loval_phase
    #         wgcoeff = wg_amps * np.exp(1j*wg_phases)
    #         input_smvals.append(wgcoeff)
    #         output_vals = mat @ wgcoeff
    #         output_modecoeffs.append(output_vals)
    #         if k % 10000 == 0:
    #             print('Making sample %d' % k)
    #
    #     return input_smvals, output_modecoeffs

































