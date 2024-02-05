import poppy
import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

import logging
# logging.basicConfig(level=logging.DEBUG)

class ZernikePSF:


    def __init__(
        self, 
        radius=1.0,
        wavelength=1500e-9,
        pixscale=0.01,
        FOV_pixels=128):

        self.psf = None
        self.wf = None
        self.radius = radius
        self.wavelength = wavelength
        self.pixscale = pixscale
        # self.FOV = 1
        self.FOV_pixels = FOV_pixels
        self.osys_obj = None
        self.pupil_wf = None


    def makeZernikePSF(
        self,
        coeffs=(0, 0, 0, 0, 0),
        show=False,
        units='microns',
        extraPlots=False):

        # RADIUS = 1.0 # meters
        # WAVELENGTH = 1500e-9 # meters
        # PIXSCALE = 0.01 # arcsec / pix
        # FOV = 1 # arcsec
        # NWAVES = 1.0
        # FOV_PIXELS = 128

        if units is 'microns':
            coeffs = np.asarray(coeffs) * 1e-6

        osys = poppy.OpticalSystem()
        circular_aperture = poppy.CircularAperture(radius=self.radius)
        osys.add_pupil(circular_aperture)
        thinlens = poppy.ZernikeWFE(radius=self.radius, coefficients=coeffs)
        osys.add_pupil(thinlens)
        #osys.add_detector(pixelscale=self.pixscale, fov_arcsec=self.FOV)
        osys.add_detector(pixelscale=self.pixscale, fov_pixels=self.FOV_pixels)

        if extraPlots:
            plt.figure(1)
        # psf_with_zernikewfe, final_wf = osys.calc_psf(wavelength=self.wavelength, display_intermediates=show,
        #                                               return_final=True)
        psf_with_zernikewfe, all_wfs = osys.calc_psf(wavelength=self.wavelength, display_intermediates=show,
                                                      return_intermediates=True)
        final_wf = all_wfs[-1]
        pupil_wf = all_wfs[1]

        if extraPlots:
            psf = psf_with_zernikewfe
            psfImage = psf[0].data
            plt.figure(2)
            plt.clf()
            poppy.display_psf(psf,
                              normalize='peak',
                              cmap='viridis',
                              scale='linear',
                              vmin=0,
                              vmax=1)
            plt.pause(0.001)
            plt.figure(3)

            wf = final_wf
            wf = pupil_wf

            plt.clf()
            plt.pause(0.001)
            plt.subplot(1, 2, 1)
            plt.imshow(wf.amplitude ** 2)
            plt.title('Amplitude ^2')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(wf.phase)
            plt.title('Phase')
            plt.colorbar()
            plt.tight_layout()

        self.psf = psf_with_zernikewfe
        self.wf = final_wf
        self.osys_obj = osys
        self.pupil_wf = pupil_wf


    def saveToRSoft(
        self,
        outfile='PSFOut',
        size_data=400):
        """
        From Theo's FITS2rsoft script.
        - The parameter 'size_data' is the physical size of the array in um so make
        sure you have the dimensions correct.

        - The format of the .fld file is 2 columns for each single column in the
        fits files, the real then the imaginary. There is a header at the top of the
        file which shouldn't need changing, it is just how the data is interpreted
        by the program
        """

        complex_psf = self.wf.amplitude * np.exp(1j * self.wf.phase)
        psf_real = complex_psf.real
        psf_imag = complex_psf.imag

        len_data = psf_real.shape[0]

        whole_psf = np.zeros([len_data, 2 * len_data])  # Empty array to take all data 3d

        # RSoft takes data in columns of real and imaginary parts. (make odd ones imag, even real)
        whole_psf[:, ::2] = psf_real
        whole_psf[:, 1::2] = psf_imag
        whole_psf = whole_psf / whole_psf.max()

        # -------------------------- Writing FLD file ----------------------------------
        outfile = outfile + '_inputfield'
        print('Writing field to file ' + outfile + '.fld')
        header = ('/rn,a,b/nx0\n/rn,qa,qb\n{0} -{1} {1} 0 OUTPUT_REAL_IMAG_3D\n{0} -{1} {1}').format(len_data, size_data)
        np.savetxt(outfile + '.fld', whole_psf, fmt='%.18E',
                   header=header, comments='')


    def makeMultiplePSFs(
        self,
        coeffsList,
        outpath='./',
        filePrefix='zernikePSFs',
        extraPlots=False,
        size_data=400,
        makeBatFile=False,
        saveAllData=False,
        indFile='bptmp.ind',
        outPrefix='BPScan',
        numBatfiles=1,
        trimLeadingCoeffnames=None, 
        wlList=None,
        wlToPoppy=True):

        allOutfiles = []
        allData = []
        if wlList is not None:
            if len(wlList) != len(coeffsList):
                print('Error: length of wlList must be same as coeffsList')
                return
        count = 0
        for coeffs in coeffsList:
            print('Making PSF with coeffs ' + str(coeffs))
            if (wlList is not None) & wlToPoppy:
                self.wavelength = wlList[count]*1e-6
            self.makeZernikePSF(coeffs=coeffs, show=True, extraPlots=extraPlots)
            plt.pause(0.001)
            # filestr = outpath + filePrefix
            filestr = filePrefix
            cocount = 0
            for c in coeffs:
                if trimLeadingCoeffnames is None:
                    filestr = filestr + '_' + '%.4f' % c
                else:
                    if cocount >= trimLeadingCoeffnames:
                        filestr = filestr + '_' + '%.4f' % c
                    cocount = cocount + 1
            if wlList is not None:
                filestr = filestr + '_' + '%.3f' % wlList[count]
            #print(filestr)
            self.saveToRSoft(outfile=outpath+filestr, size_data=size_data)
            allOutfiles.append(filestr)
            if saveAllData:
                psf_ampl = self.wf.amplitude
                psf_phase = self.wf.phase
                pupil_phase = self.pupil_wf.phase
                cur_wf = [psf_ampl, psf_phase, pupil_phase]
                allData.append(cur_wf)
            count = count + 1

        if makeBatFile:
            allOutfilenames = []
            progname = 'bsimw32'

            nf = len(allOutfiles)
            batfileLength = nf // numBatfiles
            allBatfileNames = []

            count = 0
            for k in range(numBatfiles):
                startInd = k*batfileLength
                endInd = (k+1)*batfileLength
                if k == (numBatfiles-1):
                    curOutfiles = allOutfiles[startInd:]
                else:
                    curOutfiles = allOutfiles[startInd:endInd]
                print('Making .bat file: '+outpath+outPrefix+'_'+str(k)+'.bat')
                batfile = open(outpath+outPrefix+'_'+str(k)+'.bat', 'w')
                allBatfileNames.append(outPrefix+'_'+str(k)+'.bat')
                for launch_file in curOutfiles:
                    if wlList is None:
                        cmdStr = progname + ' ' + indFile + ' prefix=' + outPrefix + launch_file + \
                                 ' launch_file=' + launch_file + '_inputfield' + '.fld wait=0\n'
                    else:
                        cmdStr = progname + ' ' + indFile + ' free_space_wavelength=' + '%.3f' % wlList[count] + \
                            ' prefix=' + outPrefix + launch_file + \
                            ' launch_file=' + launch_file + '_inputfield' + '.fld wait=0\n'
                    print(cmdStr)
                    batfile.write(cmdStr)
                    allOutfilenames.append(outPrefix + launch_file)
                    count = count + 1
                batfile.close()
            np.savez(outpath+outPrefix+'_metadata.npz', allOutfilenames=allOutfilenames, coeffsList=coeffsList,
                     allData=allData)

            if numBatfiles > 1:
                superbatfile = open(outpath+'runAllBatfiles.bat', 'w')
                for batfilename in allBatfileNames:
                    cmdStr = 'start cmd /k call ' + batfilename +'\n'
                    superbatfile.write(cmdStr)
                superbatfile.close()


    def makeCoeffsRange(
        self,
        input=None):
    
        # Input is tuple (no., start, end, numsteps)
        if input is None:
            print('0: Piston')
            print('1: Tip')
            print('2: Tilt')
            print('3: Defocus')
            print('4: Astig 1')
            print('5: Astig 2')
            print('6: Coma 1')
            print('7: Coma 2')
            print('8: Trefoil 1')
            print('9: Trefoil 2')
            print('10: Spherical')
            return

        nsteps = input[3]
        curCoeff = np.linspace(input[1], input[2], nsteps)
        coeffsList = []
        maxCoeffs = input[0] + 1
        for k in range(input[3]):
            cur = np.zeros(maxCoeffs)
            cur[input[0]] = curCoeff[k]
            coeffsList.append(cur)

        return(coeffsList)





