import numpy as np
import copy
from scipy.interpolate import interp2d
import sncosmo
from astropy.table import Table
from schwimmbad import MultiPool
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# plotting style
plt.style.use('paper.mplstyle')
twocol = 7.1014
onecol = 3.35


class Bandpass:
    """
    Class defining a bandpass filter
    
    dlamdbda is the filter wavelength resolution in angstroms
    T is the system response function
    R is the normalized response function, assuming the detector is photon-counting
    """
    
    def __init__(self, filename, name=None, dlambda=10):
        
        # load system response from file
        wavelen, T = np.loadtxt(filename, unpack=True)
        
        # resample wavelen and calculate R
        self.wavelen = np.arange(min(wavelen), max(wavelen)+dlambda, dlambda)
        self.T = np.interp(self.wavelen, wavelen, T, left=0, right=0)
        self.R = self.T * self.wavelen
        self.R /= (self.R * dlambda).sum()
        del wavelen, T
        
        # calculate mean wavelength and effective width
        self.mean_wavelen = (self.wavelen * self.R * dlambda).sum()
        self.eff_width = (self.R * dlambda).sum() / max(self.R)
        
        # set the name
        self.name = filename.split('/')[-1].split('.')[0] if name is None else name
        
    def __repr__(self):
        return 'Bandpass(' + self.name + ')'


        
class Bandpasses:
    """
    Class defining a set of Bandpass objects
    
    Loads the bandpasses listed in filter_loc/filter_list
    dlambda is the wavelength resolution of the filters in angstroms
    
    Methods without an 's' (e.g. self.band) are meant to take a single filter name
    Methods with an 's' (e.g. self.bands) are meant to take an iterable of filter names
    """
    
    def __init__(self, filter_loc='filters/', filter_list='filters.list', dlambda=10):
        names, files = np.loadtxt(filter_loc+filter_list, unpack=True, dtype=str)
        self.names = names
        self.dict = dict()
        self.dlambda = dlambda
        for name,file in zip(names,files):
            self.dict[name] = Bandpass(filter_loc+file, name, dlambda)
            
    def band(self, name):
        return self.dict[name]
    
    def bands(self, names=None):
        names = self.names if names is None else names
        return np.array([self.dict[name] for name in names])
    
    def mean_wavelen(self, name):
        return self.dict[name].mean_wavelen
    
    def mean_wavelens(self, names=None):
        names = self.names if names is None else names
        return np.array([self.dict[name].mean_wavelen for name in names])
    
    def eff_width(self, name):
        return self.dict[name].eff_width
    
    def eff_widths(self, names=None):
        names = self.names if names is None else names
        return np.array([self.dict[name].eff_width for name in names])
    
    def __repr__(self):
        return 'Bandpasses = {' + str(list(self.dict.keys()))[1:-1] + '}'



class Sed:
    """
    docstring
    """
    
    def __init__(self, wavelen=None, flambda=None, z=None):
        self.wavelen = wavelen
        self.flambda = flambda
        self._z = z
        
    def null(self):
        self.flambda = 0. * self.wavelen

    def regrid(self, wavelen):
        self.flambda = np.interp(wavelen, self.wavelen, self.flambda, left=0, right=0)
        self.wavelen = wavelen
        
    def redshift(self, z=None):
        if z is None:
            return self._z
        else:
            z0 = 0 if self._z is None else self._z
            self.wavelen = (1 + z)/(1 + z0) * self.wavelen
            self._z = z
        
    def flux(self, band):
        y = np.interp(band.wavelen, self.wavelen, self.flambda, left=0, right=0)
        flux = (y * band.R).sum() * (band.wavelen[1] - band.wavelen[0])
        return flux
    
    def fluxes(self, bandpasses, filters=None):
        filters = bandpasses.names if filters is None else filters
        return np.array([self.flux(bandpasses.band(name)) for name in filters])
    
    def train(self, observations, bandpasses, fit_bias=False, return_model=False, verbose=False, Ncpus=1):

        # begin with binning at bandpass resolution
        dlambda = bandpasses.dlambda 
        initbins = np.arange(1000, 11000+dlambda, dlambda)

        # create objects for training
        R = np.zeros(len(initbins)) # empty row that is removed after assembling R
        fluxes = np.array([])
        sigmas = np.array([])
        filters = np.array([])

        for obj in observations:

            # append bandpasses to R
            filters_ = obj.photometry['filter']
            rn = np.array([rebin_pdf(band.wavelen/(1+obj.specz),band.R*(1+obj.specz),initbins) for band in bandpasses.bands(filters_)])
            R = np.vstack((R, rn))

            # append fluxes
            fluxes = np.concatenate((fluxes, obj.photometry['flux']))
            
            # append flux errors
            sigmas = np.concatenate((sigmas, obj.photometry['flux_err']))

            # append filters
            filters = np.concatenate((filters, obj.photometry['filter']))
            
        # cut out the extraneous zeros on the wavelength tails
        idx = np.where(np.sum(R,axis=0) > 0)[0]
        idxmin, idxmax = idx[0], idx[-1] + 1
        initbins = initbins[idxmin:idxmax]
        R = R[1:,idxmin:idxmax] # we also remove first empty row of R 

        # calculate g
        g = fluxes - R @ np.interp(initbins, self.wavelen, self.flambda) * dlambda
        
        # cross validation ridge regression
        alphas = np.linspace(1e-5,1,1000)
        N_EDBs = np.arange(30,45,1)
        max_widths = np.append(np.arange(500,1100,100), None)
        kfolds = min(5,len(g))
        model = RidgeDEDB(alphas=alphas, initbins=initbins)
        cv = GridSearchCV(model, {'N_EDB':N_EDBs, 'max_width':max_widths}, cv=kfolds, n_jobs=Ncpus)
        cv.fit(R, g, sigmas=sigmas)
        model = cv.best_estimator_

        #model = RidgeDEDB(alphas=[0.6897], initbins=initbins, N_EDB=43, max_width=500)
        #model.fit(R, g, sigmas=sigmas)

        pert = model.coef_

        # if fitting for bias, perform expectation maximization
        if fit_bias:

            biases = np.zeros(len(bandpasses.names))
            filter_dict = {name:i for i,name in enumerate(bandpasses.names)}

            sed = self.copy()
            sed.regrid(model.allbins_)

            pert0, biases0 = pert * 100, biases + 100

            em_count = 0

            while not all(np.isclose([*pert,*biases], [*pert0,*biases0], rtol=1e-3, atol=1e-3)):

                em_count += 1
                print(em_count)

                pert0, biases0 = pert, biases

                # create G and g
                G = model.R_dlambda_ * (1 + biases[np.vectorize(filter_dict.get)(filters)]).reshape(-1,1)
                g = fluxes - G @ sed.flambda

                # determine the perturbation
                alphas = np.linspace(1e-5,1,1000)
                model_ = RidgeCV(alphas=alphas, fit_intercept=False)
                model_.fit(G, g, 1/sigmas**2)
                pert = model_.coef_
                alpha = model_.alpha_

                # create H and h
                h0 = model.R_dlambda_ @ (sed.flambda + pert)
                H = np.zeros((len(h0), len(bandpasses.names)))
                H[range(len(h0)), np.vectorize(filter_dict.get)(filters)] = h0
                h = fluxes - h0

                # remove the reference band bc we set its bias = 0
                ref_band = 'lssti'
                idx = filter_dict[ref_band]
                H = np.delete(H,idx,1)
                H = H[~(H==0).all(1)]
                h = h[filters != ref_band]
                sigmas_ = sigmas[filters != 'lssti']

                # determine biases
                betas = np.geomspace(1e-2, 1e8, 1000)
                model_ = LassoCV(alphas=betas, fit_intercept=True, n_jobs=Ncpus)
                model_.fit(H, h)
                biases = model_.coef_
                beta = model_.alpha_

                biases = np.insert(biases,idx,0)
                print(beta)
                print(biases)

            print(em_count)
        
        # add perturbation to original SED
        finalbins = np.append(model.allbins_, initbins[-1])
        pert = np.append(pert, 0)
        self.flambda += np.interp(self.wavelen, finalbins, pert, left=0, right=0)

        return h,h0[filters != ref_band],filters[filters != ref_band]

        # print statements
        names = ['alpha', 'N_EDB', 'Max width', 'N_split']
        vals = [round(model.alpha_,4), model.N_EDB, model.max_width, model.Nsplit_]
        cv_ranges = [alphas, N_EDBs, max_widths[:-1]]
        
        if verbose:
            for name,val in zip(names,vals):
                print(f"{name} = {val}")
            if fit_bias:
                print(f'beta = {beta:.4f}')
                print('Biases:')
                for name,bias in zip(bandpasses.names,biases):
                    print(f'{name}: {bias:>7.4f}')
        
        for name, val, cv_range in zip(names[:-1],vals[:-1],cv_ranges):
            if val == min(cv_range):
                print(f"Warning: {name} = {val}, which is the minimum of the tested range.\n",
                        f"Consider lowering the range of {name}s tested.")
            elif val == max(cv_range):
                print(f"Warning: {name} = {val}, which is the maximum of the tested range.\n",
                        f"Consider raising the range of {name}s tested.")
                

        if return_model:
            return model, biases

    def copy(self):
        return copy.deepcopy(self)

    
class RidgeDEDB(BaseEstimator):
    
    def __init__(self, N_EDB=40, alphas=np.linspace(1e-3,100,1000),
                 initbins=np.arange(1000, 11010, 10), max_width=None):
        self.N_EDB = N_EDB
        self.alphas = alphas
        self.initbins = initbins
        self.max_width = max_width
        
    def fit(self, R, g, sigmas=None):
        
        # if errors aren't provided, weight all photometry equally
        sigmas = np.ones(len(g)) if sigmas is None else sigmas
        
        # calculate info density and cumulative info
        dlambda = self.initbins[1] - self.initbins[0]
        infoDen = np.sum(R, axis=0)/(np.sum(R) * dlambda)
        self.infoDen_ = infoDen
        cumInfo = np.cumsum(infoDen * dlambda)
        self.cumInfo_ = cumInfo
        
        # dynamically determine the equal density bins
        infobins = np.linspace(0.001, 0.999, self.N_EDB)
        self.infobins_ = infobins
        EDbins = np.interp(infobins, cumInfo, self.initbins, left=0, right=0)
        self.EDbins_ = EDbins
        
        # split any bins larger than max_width
        allbins = self.split_bins()
        self.allbins_ = allbins

        # rebin R and calculate R * dlambda
        R_dlambda = np.array([rebin_pdf(self.initbins, row, allbins) * np.diff(allbins, append=0) for row in R])
        self.R_dlambda_ = R_dlambda

        # perform cross-validated Ridge Regression on alpha
        model = RidgeCV(alphas=self.alphas, fit_intercept=False)
        model.fit(R_dlambda, g, 1/sigmas**2)
        
        self.Nsplit_ = len(allbins) - self.N_EDB
        self.alpha_ = model.alpha_
        self.coef_ = model.coef_
        
        return self
    
    def predict(self, R):

        R_dlambda = np.array([rebin_pdf(self.initbins, row, self.allbins_) * np.diff(self.allbins_, append=0) for row in R])
        
        return R_dlambda @ self.coef_
    
    def score(self, R, g):
        
        u = ((g - self.predict(R))**2).sum()
        v = ((g - g.mean())**2).sum()
        
        return 1 - u/v

    def split_bins(self):

        bins = self.EDbins_.copy()
        if self.max_width is None:
            return bins
        else:
            diffs = np.diff(bins)
            while any(diffs > self.max_width):
                idx = np.where(diffs > self.max_width)[0][0]
                bins = np.insert(bins, idx+1, 1/2*(bins[idx] + bins[idx+1]))
                diffs = np.diff(bins)

            return bins    


def rebin_pdf(x, y, bins):
    pdf = np.interp(bins, x, y, left=0, right=0)
    norm = np.sum(pdf * np.diff(bins,append=0))
    return np.zeros(len(bins)) if norm == 0 else pdf/norm


class LightCurve:
    """
    docstring
    """
    
    def __init__(self, time=None, wavelen=None, flambda=None, z=None):
        self._time = time
        self._wavelen = wavelen
        self.flambda = flambda
        self._z = z

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time):
        self._time = time
        if self.time is None or self.wavelen is None or self.flambda is None:
            self._flux = None
        else:
            self._flux = interp2d(self.time, self.wavelen, self.flambda)

    @property
    def wavelen(self):
        return self._wavelen

    @wavelen.setter
    def wavelen(self, wavelen):
        self._wavelen = wavelen
        if self.time is None or self.wavelen is None or self.flambda is None:
            self._flux = None
        else:
            self._flux = interp2d(self.time, self.wavelen, self.flambda)

    @property
    def flambda(self):
        return self._flambda

    @flambda.setter
    def flambda(self, flambda):
        self._flambda = flambda
        if self.time is None or self.wavelen is None or self.flambda is None:
            self._flux = None
        else:
            self._flux = interp2d(self.time, self.wavelen, self.flambda)

    def redshift(self, z=None):
        if z is None:
            return self._z
        else:
            z0 = 0 if self._z is None else self._z
            self.wavelen = (1 + z)/(1 + z0) * self.wavelen
            self._z = z

    @property
    def tmin(self):
        return None if self.time is None else min(self.time)
    
    @property
    def tmax(self):
        return None if self.time is None else max(self.time)
    
    @property
    def wmin(self):
        return None if self.wavelen is None else min(self.wavelen)
    
    @property
    def wmax(self):
        return None if self.wavelen is None else max(self.wavelen)
            
    def flux(self, time, wavelen):
        return None if self._flux is None else np.squeeze( self._flux(time, wavelen) )
    
    def bandflux(self, time, band):
        sed = Sed(self.wavelen, self.flux(time, self.wavelen))
        return sed.flux(band)
        
    def bandfluxes(self, time, bandpasses, filters=None):
        time = np.array([time]) if '__iter__' not in dir(time) else time
        seds = [Sed(self.wavelen, self.flux(t, self.wavelen)) for t in time]
        flambda = np.array([sed.fluxes(bandpasses,filters) for sed in seds])
        return np.squeeze(flambda)

    def null(self):
        self.flambda = np.zeros( (len(self.wavelen), len(self.time)) )
        
    def from_model(self, model, norm=10):
        z = 0 if self._z is None else self._z
        wavelen = self.wavelen/(1 + z)
        flambda = model.flux(self.time, wavelen)
        norm = np.max(flambda) if norm is None else norm
        self.flambda = flambda.T * norm/np.max(flambda)
    
    def regrid(self, time, wavelen):
        self.flambda = self.flux(time,wavelen)
        self.time = time
        self.wavelen = wavelen
    
    def sed_slice(self, time):
        return Sed(self.wavelen, self.flux(time,self.wavelen))
    
    def sed_slices(self):
        return {t:Sed(self.wavelen, self.flux(t,self.wavelen)) for t in self.time}

    def training_sets(self, observations):

        training_sets = {t:[] for t in self.time}

        for obj in observations:
            sort_dict = {t:[] for t in self.time}
            time = np.array(obj.photometry['mjd'] - obj.t0)/(1 + obj.specz)

            for i,t in enumerate(time):
                T = self.time[np.abs(self.time - t).argmin()]
                sort_dict[T].append(i)

            for t,idx in sort_dict.items():
                if len(idx) > 0:
                    obj_ = copy.deepcopy(obj)
                    obj_.photometry = obj_.photometry[idx]
                    training_sets[t].append(obj_)

        return training_sets

    def train(self, training_sets, bandpasses, Ncpus=None, verbose=False):
        
        sedslices = self.sed_slices()
        trained_flambda = []

        for t,sed in sedslices.items():
            if verbose:
                print(t)
            sed.train(training_sets[t], bandpasses, Ncpus=Ncpus)
            trained_flambda.append(sed.flambda)
        
        self.flambda = np.array(trained_flambda).T

    def copy(self):
        return copy.deepcopy(self)
    
    def surface_plot(self, figsize=(twocol,twocol), cmap='viridis'):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        
        x,y = np.meshgrid(self.time, self.wavelen)
        surf = ax.plot_surface(x, y, self.flambda, cmap=cmap, rcount=200, ccount=200)
        
        plt.setp( ax.xaxis.get_majorticklabels(), va="bottom", ha='right' )
        plt.setp( ax.yaxis.get_majorticklabels(), va="bottom", ha='left' )
        ax.set_xlabel("Phase (Days)", ha='left')
        ax.set_ylabel("Wavelength ($\mathrm{\AA}$)", labelpad=10)
        ax.set_zlabel("Flux Density", labelpad=2)
        ax.view_init(30, -60)

        return fig, ax
    
    def contour_plot(self, figsize=(onecol,onecol), cmap='viridis'):
        fig,ax = plt.subplots(1,1, figsize=figsize, constrained_layout=True)
        
        x,y = np.meshgrid(self.time, self.wavelen)

        zcut = np.max(self.flambda)/20
        z = np.log10( np.clip(self.flambda, zcut, None) )
        ax.contourf(x, y, z, levels=200)
        ax.set_xlabel("Phase (Days)")
        ax.set_ylabel("Wavelength ($\mathrm{\AA}$)")
        
        return fig, ax

    def stacked_plot(self, trange=None, tstep=5, offset_scale=1):

        trange = np.arange(0, self.tmax, tstep) if trange is None else trange

        fig,ax = plt.subplots()

        for i,t in enumerate(trange):
            sed = self.sed_slice(t)
            idx = np.abs(sed.wavelen - 10000).argmin()
            ax.plot(sed.wavelen[:idx], sed.flambda[:idx] + len(trange) - i*offset_scale, c='k')
            ax.text(sed.wavelen[idx], sed.flambda[idx] + len(trange) - i*offset_scale, t, ha='left', va='center')

        ax.set_xlabel("Wavelength ($\mathrm{\AA}$)")
        ax.set_ylabel("Flux Density + Offset")
        ax.set_xlim(1200,10800)
        ax.set_yticks([])

        return fig, ax


class SkyObject:
    """
    Class defining an object observed in the sky.
    
    photometry is an astropy table describing the observations of the object
    source is the type of astrophysical source, e.g. SN1a
    t0 is a reference time, e.g. mjd of peak flux for a SN1a
    Nobs() returns the number of observations of the object
    """
    
    def __init__(self, photometry=None, specz=None, photoz=None, photoz_err=None, source=None, t0=None, distmod=None):
        self.photometry = photometry
        self.specz = specz
        self.photoz = photoz
        self.photoz_err = photoz_err
        self.source = source
        self.t0 = t0
        self.distmod = distmod
    
    @property
    def Nobs(self):
        return len(self.photometry) if '__len__' in dir(self.photometry) else None
        
    def __repr__(self):
        string = ( 'SkyObject Observation: \n\n'
                  f'{"source":>11} = {str(self.source):<6} \n'
                  f'{"t0":>11} = {str(self.t0):<6} \n'
                  f'{"spec-z":>11} = {str(self.specz):<6} \n'
                  f'{"photo-z":>11} = {str(self.photoz):<6} \n'
                  f'{"photo-z err":>11} = {str(self.photoz_err):<6} \n'
                  f'{"N obs":>11} = {str(self.Nobs):<6} \n\n' ) + \
                  '\n'.join(self.photometry.pformat(max_lines=16))
        
        return string



class SNSurvey:
    """
    Photometric survey of supernovae
    
    obs is the array of photometric observations, each of which is an astropy table
    model is the sncosmo model for simulating the survey
    zmin and zmax are the min and max redshifts
    area is the area of the survey in sq. deg.
    duration is the duration of the survey in days
    cadence how often observations are made of each object. In the simulations, flux is measured
            in one filter at a time, starting with a random filter and cycling through
    flux_errf is the fractional error of the fluxes. Fluxes are calculated with Gaussian noise.
    the self.simulate(bandpasses) method simulates a photometric SN survey with all of these parameters
    """
    
    def __init__(self, obs=None, model=None, zmin=0, zmax=1, area=1, duration=1e3, cadence=1, 
                 flux_errf=0.05, bias=None, norm=None):
        self.obs = obs
        self.model = model
        self.zmin = zmin
        self.zmax = zmax
        self.area = area
        self.duration = duration
        self.cadence = cadence
        self.flux_errf = flux_errf
        self.bias = bias

    @property
    def Nobs(self):
        return len(self.obs) if '__len__' in dir(self.obs) else None
        
    def simulate(self, bandpasses, norm=None, seed=13, Ncpus=None):
        
        self.obs = np.array([])
        
        np.random.seed(seed)
        
        tstep = 1
        tmin = self.model.mintime()
        tmax = self.model.maxtime()
        time = np.arange(tmin, tmax + tstep, tstep)
        
        wstep = 10
        wmin = self.model.minwave()
        wmax = self.model.maxwave()
        wavelen = np.arange(wmin, wmax + wstep, wstep)
        
        fluxes = self.model.flux(time, wavelen)
        norm = np.max(fluxes) if norm is None else norm
        fluxes = fluxes.T * norm/np.max(fluxes)
        
        lc = LightCurve(time, wavelen, fluxes)
        
        redshifts = list(sncosmo.zdist(self.zmin, self.zmax, time=self.duration, area=self.area))

        tasks = list(zip(redshifts, [self]*len(redshifts), [bandpasses]*len(redshifts),
                    [self.bias]*len(redshifts), [lc]*len(redshifts), 
                    np.random.randint(2**32 - 1,size=len(redshifts))))
        with MultiPool(processes=Ncpus) as pool:
            observations = np.array(list(pool.map(survey_worker, tasks)))
        self.obs = observations
        
        
            

            
    def __repr__(self):
        string = ( 'SN Survey Simulation: \n\n'
                  f'{"N obs":>9} = {str(self.Nobs):<6} \n'
                  f'{"zmin":>9} = {str(self.zmin):<6} \n'
                  f'{"zmax":>9} = {str(self.zmax):<6} \n'
                  f'{"area":>9} = {str(self.area):<6} \n'
                  f'{"duration":>9} = {str(self.duration):<6} \n'
                  f'{"cadence":>9} = {str(self.cadence):<6} \n'
                  f'{"flux errf":>9} = {str(self.flux_errf):<6} \n'
                  f'{"bias":>9} = {str(self.bias)} \n\n' ) + \
                  'Model: \n' + '\n'.join(self.model.__str__().split('\n')[1:])
        
        return string


def survey_worker(task):

    z = task[0]
    survey = task[1]
    bandpasses = task[2]
    bias = task[3]
    lc = task[4].copy()
    lc.time *= (1 + z)
    tmin = lc.tmin
    tmax = lc.tmax
    seed = task[5]

    np.random.seed(seed)

    t0 = 59600 + np.random.rand() * (survey.duration - tmax)
    toffset = survey.cadence * np.random.rand()
    filter_order = np.roll( bandpasses.names, np.random.randint(len(bandpasses.names)) )
    
    mjd = np.array([])
    fluxes = np.array([])
    flux_errs = np.array([])
    filters = np.array([])
    filter_dict = {name:i for i,name in enumerate(bandpasses.names)}
    
    for i,t in enumerate( np.arange(tmin + toffset, tmax, survey.cadence) ):
        
        tscatter = 0.5*np.random.rand() - 0.25
        T = t + tscatter
        mjd = np.append(mjd, round(t0 + T, 4))
        
        band_name = filter_order[ i % len(filter_order) ]
        band = bandpasses.band(band_name)
        filters = np.append(filters, band_name)
        
        sed = lc.sed_slice(T)
        sed.redshift(z)
        
        flux = sed.flux(band)
        flux *= (1 + bias[filter_dict[band_name]])
        flux_err = np.clip(np.fabs(survey.flux_errf * flux), 1e-4, None)
        flux = round(np.random.normal(flux, flux_err), 6)
        flux_err = np.clip(round(flux_err, 6), 1e-6, None)

        fluxes = np.append(fluxes, flux)
        flux_errs = np.append(flux_errs, flux_err)
    
    observation = SkyObject()
    observation.specz = round(z, 4)
    observation.source = 'SN1a'
    observation.t0 = round(t0, 4)
    observation.photometry = Table( data=(mjd, filters, fluxes, flux_errs),
                                    names=('mjd', 'filter', 'flux', 'flux_err') )
    
    return observation