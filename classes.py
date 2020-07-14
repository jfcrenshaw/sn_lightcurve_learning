import numpy as np
import copy
from scipy.interpolate import interp2d
import sncosmo
from astropy.table import Table
from schwimmbad import MultiPool
from sklearn.linear_model import RidgeCV
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
        self.T = np.interp(self.wavelen, wavelen, T)
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
        
    def redshift(self, z=None):
        if z is None:
            return self._z
        else:
            z0 = 0 if self._z is None else self._z
            self.wavelen = (1 + z)/(1 + z0) * self.wavelen
            self._z = z
        
    def flux(self, band):
        y = np.interp(band.wavelen, self.wavelen, self.flambda)
        flux = (y * band.R).sum() * (band.wavelen[1] - band.wavelen[0])
        return flux
    
    def fluxes(self, bandpasses, filters=None):
        filters = bandpasses.names if filters is None else filters
        return np.array([self.flux(bandpasses.band(name)) for name in filters])
    
    def train(self, observations, bandpasses, return_all=False):

        dlambda = bandpasses.dlambda 
        bins = np.arange(1000, 11000+dlambda, dlambda)

        sigmas = np.array([])
        R = np.zeros(len(bins)-1)
        g = np.array([])

        for obj in observations:

            filters = obj.photometry['filter']
            rn = np.array([np.histogram(band.wavelen/(1 + obj.specz), weights=band.R, bins=bins, density=True)[0] for band in bandpasses.bands(filters)])
            R = np.vstack((R, rn))

            sed = self.copy()
            sed.redshift(obj.specz)
            g = np.concatenate((g, obj.photometry['flux'] - sed.fluxes(bandpasses, filters)))
            sigmas = np.concatenate((sigmas, obj.photometry['flux_err']))
        
        infoDen = np.sum(R, axis=0)/(np.sum(R)*dlambda)
        idx = np.where(infoDen > 0)[0]
        bins, infoDen = bins[idx], infoDen[idx]
        cumInfo = np.cumsum(infoDen*dlambda)
        cumInfo[0], cumInfo[-1] = 0, 1   

        R = R[1:,idx]

        # first do a broad search of N
        N = np.arange(30,70,10)
        alphas = np.linspace(1e-5,100,1000)
        model = RidgeDEDB(cumInfo=cumInfo, alphas=alphas, bins=bins)
        clf = GridSearchCV(model, {'N':N})
        clf.fit(R, g, sigmas=sigmas)
        model = clf.best_estimator_

        # now do a finer search of N
        N_approx = model.N 
        N = np.arange(N_approx-5, N_approx+6, 1)
        clf = GridSearchCV(model, {'N':N})
        clf.fit(R, g, sigmas=sigmas)
        model = clf.best_estimator_

        # now possibly split bins
        max_widths = np.arange(600,max(np.diff(model.EDbins_))+100,100)
        model = RidgeDEDB(cumInfo=cumInfo, alphas=alphas, bins=bins, N=model.N_)
        clf = GridSearchCV(model, {'max_width':max_widths})
        clf.fit(R, g, sigmas=sigmas)
        model = clf.best_estimator_

        EDbins = model.EDbins_ + 100 # NEED TO FIND WHY THIS OFFSET IT NEEDED
        pert = np.append(model.coef_, 0)

        self.flambda += np.interp(self.wavelen, EDbins, pert)

        print("N =",model.N_)
        print("Nsplit =",model.Nsplit_)
        print("max width =",model.max_width_)
        print("alpha =",model.alpha_)

        if model.alpha_ == min(alphas):
            print(f"Warning: alpha = {model.alpha_}, which is the minimum of the tested range.\n",
                    "Consider lowering the range of alphas tested.")
        elif model.alpha_ == max(alphas):
            print(f"Warning: alpha = {model.alpha_}, which is the max of the tested range.\n",
                    "Consider increasing the range of alphas tested.")

        # ALSO WRITE WARNINGS FOR MAX/MIN OF N RANGE

        if return_all:
            return EDbins, pert, bins, infoDen, cumInfo 

    def copy(self):
        return copy.deepcopy(self)


class RidgeDEDB(BaseEstimator):
    
    def __init__(self, N=40, cumInfo=None, 
                 alphas=np.linspace(1e-3,100,1000),
                 bins=np.arange(1000, 11010, 10),
                 max_width=None):

        self.N = N
        self.cumInfo = cumInfo
        self.alphas = alphas
        self.bins = bins
        self.max_width = max_width
        
    def fit(self, R, g, sigmas):
        
        infoBins = np.linspace(0.,1.,self.N)
        EDbins = np.interp(infoBins,self.cumInfo,self.bins)
        EDbins = self.split(EDbins, self.max_width) if self.max_width is not None else EDbins

        R_dlambda = np.array([np.histogram(self.bins, weights=row, bins=EDbins, density=True)[0] * np.diff(EDbins) for row in R])

        model = RidgeCV(alphas=self.alphas, fit_intercept=False)
        model.fit(R_dlambda, g, 1/sigmas**2)
        
        self.N_ = self.N
        self.Nsplit_ = len(EDbins) - self.N
        self.max_width_ = self.max_width
        self.EDbins_ = EDbins
        self.alpha_ = model.alpha_
        self.coef_ = model.coef_
        
        return self
    
    def predict(self, R):
        
        infoBins = np.linspace(0.,1.,self.N)
        EDbins = np.interp(infoBins,self.cumInfo,self.bins)
        EDbins = self.split(EDbins, self.max_width) if self.max_width is not None else EDbins

        R_dlambda = np.array([np.histogram(self.bins, weights=row, bins=EDbins, density=True)[0] * np.diff(EDbins) for row in R])
        
        return R_dlambda @ self.coef_
    
    def score(self, R, g):
        
        u = ((g - self.predict(R))**2).sum()
        v = ((g - g.mean())**2).sum()
        
        return 1 - u/v

    def split(self,bins,max_width):

        bins_ = bins.copy()

        diffs = np.diff(bins)
        while any(diffs > max_width):
            idx = np.where(diffs > max_width)[0][0]
            bins_ = np.insert(bins_, idx+1, 0.5*(bins_[idx]+bins_[idx+1]))
            diffs = np.diff(bins_)

        return bins_
    


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

    def mse(self, training_sets, bandpasses, Ncpus=None):

        sedslices = self.sed_slices()
        
        tasks = list(zip(sedslices.values(), training_sets.values(), [bandpasses]*len(sedslices)))
        with MultiPool(processes=Ncpus) as pool:
            results = np.array(list(pool.map(mse_worker, tasks)))
        N = sum([i[0] for i in results])
        se = sum([i[1] for i in results])
        mse = se/N if N > 0 else 0
        return mse

    def perturb(self, training_sets, bandpasses, w=10, Delta=None, Ncpus=None):
            
        sedslices = self.sed_slices()
        keys = np.array(list(sedslices.keys()))
        
        tasks = list(zip(sedslices.values(), training_sets.values(), 
                         [bandpasses]*len(keys), [w]*len(keys), [Delta]*len(keys)))
        with MultiPool(processes=Ncpus) as pool:
            newflambda = np.array(list(pool.map(perturbation_worker, tasks)))
            
        self.flambda = newflambda.T

    def train(self, training_sets, bandpasses, w=10, Delta=None, 
                dmse_stop=0.03, maxPerts=None, Ncpus=None):
        
        sedslices = self.sed_slices()
        keys = np.array(list(sedslices.keys()))
            
        tasks = list(zip(sedslices.values(), training_sets.values(), 
                         [bandpasses]*len(keys), [w]*len(keys), [Delta]*len(keys),
                         [dmse_stop]*len(keys), [maxPerts]*len(keys)))
        with MultiPool(processes=Ncpus) as pool:
            newflambda = np.array(list(pool.map(training_worker, tasks)))
            
        self.flambda = newflambda.T

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


def mse_worker(task):
    sed = task[0]
    training_set = task[1]
    bandpasses = task[2]
    N = 0
    for obj in training_set:
        N += len(obj.photometry['flux'])
    return [N, N*sed.mse(training_set, bandpasses)]
    
def perturbation_worker(task):
    sed = task[0]
    training_set = task[1]
    bandpasses = task[2]
    w = task[3]
    Delta = task[4]
    sed.perturb(training_set, bandpasses, w, Delta)
    return sed.flambda

def training_worker(task):
    sed = task[0]
    training_set = task[1]
    bandpasses = task[2]
    w = task[3]
    Delta = task[4]
    dmse_stop = task[5]
    maxPerts = task[6]

    if len(training_set) > 0:
        sed.train(training_set, bandpasses, w, Delta, dmse_stop, maxPerts)
        
    return sed.flambda



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
                 flux_errf=0.05, norm=None):
        self.obs = obs
        self.model = model
        self.zmin = zmin
        self.zmax = zmax
        self.area = area
        self.duration = duration
        self.cadence = cadence
        self.flux_errf = flux_errf

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
                    [lc]*len(redshifts), np.random.randint(2**32 - 1,size=len(redshifts))))
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
                  f'{"flux errf":>9} = {str(self.flux_errf):<6} \n\n' ) + \
                  'Model: \n' + '\n'.join(self.model.__str__().split('\n')[1:])
        
        return string


def survey_worker(task):

    z = task[0]
    survey = task[1]
    bandpasses = task[2]
    lc = task[3].copy()
    lc.time *= (1 + z)
    tmin = lc.tmin
    tmax = lc.tmax
    seed = task[4]

    np.random.seed(seed)

    t0 = 59600 + np.random.rand() * (survey.duration - tmax)
    toffset = survey.cadence * np.random.rand()
    filter_order = np.roll( bandpasses.names, np.random.randint(len(bandpasses.names)) )
    
    mjd = np.array([])
    fluxes = np.array([])
    flux_errs = np.array([])
    filters = np.array([])
    
    for i,t in enumerate( np.arange(tmin + toffset, tmax, survey.cadence) ):
        
        tscatter = 0.5*np.random.rand() - 0.25
        T = t + tscatter
        mjd = np.append(mjd, round(t0 + T, 4))
        
        band_name = filter_order[ i % len(filter_order) ]
        band = bandpasses.band(band_name)
        filters = np.append(filters, band_name)
        
        sed = lc.sed_slice(T)
        sed.redshift(z)
        
        flux = np.clip(sed.flux(band), 1e-3, None) # need to figure out a way to handle negative and zero fluxes
        flux_err = np.fabs(survey.flux_errf * flux)
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