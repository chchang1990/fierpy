import numpy as np
import xarray as xr
import pandas as pd
import random
import scipy as sci
import matplotlib
import matplotlib.pyplot as plt

import os
import logging
from pathlib import Path

from eofs.xarray import Eof
from geoglows import streamflow
from sklearn import metrics
from sklearn.model_selection import train_test_split
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def unrot_eof(stack: xr.DataArray, variance_threshold: float = 0.8, n_modes: int = -1) -> xr.Dataset:
    """Function to perform rotated empirical othogonal function (eof) on a spatial timeseries

    args:
        stack (xr.DataArray): DataArray of spatial temporal values with coord order of (t,y,x)
        variance_threshold(float, optional): optional fall back value to select number of eof
            modes to use. Only used if n_modes is less than 1. default = 0.727
        n_modes (int, optional): number of eof modes to use. default = 4

    returns:
        xr.Dataset: rotated eof dataset with spatial modes, temporal modes, and mean values
            as variables

    """
    # extract out some dimension shape information
    shape3d = stack.shape
    spatial_shape = shape3d[1:]
    shape2d = (shape3d[0],np.prod(spatial_shape))

    # flatten the data from [t,y,x] to [t,...]
    da_flat = xr.DataArray(
        stack.values.reshape(shape2d),
        coords = [stack.time,np.arange(shape2d[1])],
        dims=['time','space']
    )
    #logger.debug(da_flat)
        
    ## find the temporal mean for each pixel
    center = da_flat.mean(dim='time')
    
    centered = da_flat - center
               
    # get an eof solver object
    # explicitly set center to false since data is already
    #solver = Eof(centered,center=False)
    solver = Eof(centered,center=False)

    # check if the n_modes keyword is set to a realistic value
    # if not get n_modes based on variance explained
    if n_modes < 0:
        n_modes = int((solver.varianceFraction().cumsum() < variance_threshold).sum())

    # calculate to spatial eof values
    eof_components = solver.eofs(neofs=n_modes).transpose()
    # get the indices where the eof is valid data
    non_masked_idx = np.where(np.logical_not(np.isnan(eof_components[:,0])))[0]

    # # waiting for release of sklean version >= 0.24
    # # until then have a placeholder function to do the rotation
    # fa = FactorAnalysis(n_components=n_modes, rotation="varimax")
    # rotated[non_masked_idx,:] = fa.fit_transform(eof_components[non_masked_idx,:])

    # get eof with valid data
    eof_components = np.asarray(eof_components)

    
    # project the original time series data on the rotated eofs
    projected_pcs = np.dot(centered[:,non_masked_idx], eof_components[non_masked_idx,:])

    # reshape the rotated eofs to a 3d array of [y,x,c]
    spatial = eof_components.reshape(spatial_shape+(n_modes,))
    
    
    # structure the spatial and temporal eof components in a Dataset
    eof_ds = xr.Dataset(
        {
            "spatial_modes": (["lat","lon","mode"],spatial),
            "temporal_modes":(["time","mode"],projected_pcs),
            "center": (["lat","lon"],center.values.reshape(spatial_shape))
        },
        coords = {
            "lon":(["lon"],stack.lon),
            "lat":(["lat"],stack.lat),
            "time":stack.time,
            "mode": np.arange(n_modes)+1
        }
    )

    return eof_ds
    
    
def sig_eof_test(stack: xr.DataArray, option: int=1, monte_carlo_iter: int=10):

    """
    Significant test upon the EOF analysis results. The purpose is to find out the significant EOF modes
    to retain for rotation (REOF)

    args:
        stack (xr.DataArray): Input DataArray containing satellite imagery
        option (int): Type of significant test (1: Monte Carlo test; 2: North's rule-of-thumb)
        monte_carlo_iter (int): Iteration time for Monte Carlo test. Only applies when "option=1"

    output:
        sig_mode (int): number of modes to retain for the rotation of EOF analysis results

    """
    from sklearn.decomposition import PCA

    fontdict={
       'weight':'bold',
       'size':16
    }
    matplotlib.rc('font',**fontdict)

    time_size = stack.sizes['time']
    lat_size = stack.sizes['lat']
    lon_size = stack.sizes['lon']
    space_size = lat_size*lon_size

    temporal_mean = np.nanmean(stack.values, axis=0)

    flat_da = stack.values.reshape((time_size, space_size))
    aoi_mask_flat = ~(np.isnan(flat_da).all(axis=0))
    flat_da = flat_da[:, aoi_mask_flat]
    flat_da = flat_da - flat_da.mean(axis=0, keepdims=True)

    # ----- Record the original index of each pixels inside the AOI to recover the flattened data back to geographical dimension -----
    flat2geo = np.arange(np.prod(space_size))[aoi_mask_flat].reshape(1,-1)

    # ----- sklearn PCA (Eigenvalue of real data) -----
    pca = PCA(n_components=time_size)
    pcs = pca.fit_transform(flat_da)
    real_lamb = pca.explained_variance_

    if option==1:

        mc_lamb = np.full((time_size,monte_carlo_iter),np.nan)

        # ----- Monte Carlo simulation -----
        for i in range(monte_carlo_iter):

            # ----- Randomize observation (spatially shuffle the data) -----
            rng = np.random.default_rng()
            obs_temp = rng.permuted(flat_da, axis=1)

            pca = PCA(n_components=time_size)
            pcs = pca.fit_transform(obs_temp)
            eigv = pca.explained_variance_
            mc_lamb[:,i] = eigv

        mc_lamb = np.transpose(mc_lamb)
        mean_mc_lamb = np.mean(mc_lamb,axis=0)
        std_mc_lamb = np.std(mc_lamb,axis=0)

        plt.title('Scree Plot',fontdict=fontdict)
        plt.plot(np.arange(time_size)+1, real_lamb, marker='+')
        plt.errorbar(np.arange(time_size)+1, mean_mc_lamb, std_mc_lamb, capsize=1, marker='o', markersize=3)
        plt.legend(['From real data','From MonteCarlo sim.'])
        plt.xlabel('Mode',fontdict=fontdict)
        plt.ylabel('Eigenvalue',fontdict=fontdict)
        plt.show()

        sig_bool = real_lamb > mean_mc_lamb
        sig_mode = (np.argwhere(sig_bool==False)[0])[0]

    elif option==2:
        lamb_err = real_lamb*np.sqrt(2/time_size)
        lower_lamb = real_lamb - lamb_err
        upper_lamb = real_lamb + lamb_err

        real_lamb_temp = real_lamb.copy()

        sig_bool = lower_lamb[:-1] > upper_lamb[1:]
        sig_mode = (np.argwhere(sig_bool==False)[0])[0]

        plt.figure(figsize=(6,5))
        plt.errorbar(np.arange(len(real_lamb))+1, real_lamb, yerr=lamb_err, capsize=5, marker='o', markersize=3)
        plt.plot([sig_mode, sig_mode], [np.max(upper_lamb), np.min(lower_lamb)], color='r')
        plt.ylabel('Eigenvalue', fontdict=fontdict)
        plt.xlabel('Mode', fontdict=fontdict)
        plt.show()

    return sig_mode
                                                           
    
def find_hydro_mode(eof_stack: xr.Dataset, hydro_stack: xr.DataArray, r_thrd: float=0.5) -> xr.Dataset:
    """
    Calculate the correlation between temporal patterns and hydrological data.
    This helps determine the water-related mode. By default,  >=0.5 is considered to be correlated.

    args:
       eof_stack: Dataset with EOF or REOF results
       hydro_stack: DataArray with hydrological data (site-by-time)
       r_thrd: Threshold (default: 0.5) of correlation coefficient to decide which modes are water-related

    output (site X modes):
       site: Names of selected sites of the modes
       best_r: The highest correlations between RTPCs and hydrological data
       best_p: Corresponding p-values of best_r
    """
    # get number of mode
    mode_num = eof_stack.sizes['mode']
    # get number of hydrological data sites
    site_num = hydro_stack.sizes['site']

    r = np.zeros((site_num, mode_num))
    p = np.zeros((site_num, mode_num))

    time_tpc = eof_stack.temporal_modes.sel(mode=int(1)).time
    time_hydro = hydro_stack[0].time
    comm_indx_hydro = time_hydro.isin(time_tpc)
    comm_indx_tpc = time_tpc.isin(hydro_stack[:,comm_indx_hydro.values].time)
    #print(hydro_stack[:,comm_indx_hydro.values].time.values)
    hydro_site = hydro_stack.site.values
    hydro_stack = hydro_stack[:,comm_indx_hydro.values].values


    n = hydro_stack.shape[-1]
    p_dist = stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)

    # ----- Instead of using FOR, see if it is possible to vectorize the process -----
    for ct_mode in range(mode_num):
        # get mode of tpc
        tpc = eof_stack.temporal_modes.sel(mode=int(ct_mode+1))[comm_indx_tpc]
        tpc = tpc.expand_dims(dim='site',axis=0).values.reshape(1,-1)

        r_da = np.concatenate((tpc, hydro_stack), axis=0)
        r_mode = np.expand_dims((pd.DataFrame(np.transpose(r_da)).corr(method='pearson').values[1:,0]), axis=1)
        p_mode = 2*p_dist.cdf(-abs(r_mode))

        if ct_mode==0:
            r = r_mode
            p = p_mode
        else:
            r = np.concatenate((r,r_mode), axis=1)
            p = np.concatenate((p,p_mode), axis=1)

    indx_max_r_site = np.nanargmax(np.abs(r), axis=0)
    r_temp = r[( indx_max_r_site, list(range(mode_num)) )]
    p_temp = p[( indx_max_r_site, list(range(mode_num)) )]

    indx_p_sig = p_temp <= 0.05
    indx_r_sig = np.abs(r_temp) >= r_thrd
    mode_out = ((np.logical_and(np.abs(r_temp) >= r_thrd, p_temp <= 0.05).astype(int)).nonzero())[0]

    site_out = indx_max_r_site[mode_out]
    r_out = r_temp[mode_out]
    p_out = p_temp[mode_out]

    tpc_hydro_r = xr.Dataset(

        data_vars=dict(
            site=(["mode"], hydro_site[site_out]),
            best_r=(["mode"], r_out),
            best_p=(["mode"], p_out)
        ),

        coords=dict(
            mode=(["mode"], eof_stack.mode.values[mode_out]),
        ),

    )

    return tpc_hydro_r

        
        
def wrap_streamflow(lats: list, lons: list) -> Tuple[xr.DataArray, list]:
    """Function to get and wrap up histroical streamflow data from the GeoGLOWS server
    at different geographic coordinates

    args:
        lats (list): latitude values where to get streamflow data
        lons (list): longitude values where to get streamflow data

    returns:
        xr.DataArray: DataArray object of streamflow with datetime coordinates
    """
    site_num = len(lats)
    reaches = []
    for ct_site in range(site_num):
      if ct_site==0:
        q, reach_id = get_streamflow(lats[ct_site], lons[ct_site])
        q["time"] = q["time"].dt.strftime("%Y-%m-%d")
        q.expand_dims(dim='site')
      else:
        q1, reach_id = get_streamflow(lats[ct_site], lons[ct_site])
        q1["time"] = q1["time"].dt.strftime("%Y-%m-%d")        
        q = xr.concat( (q, q1),dim='site' ) 
      reaches.append(reach_id)
      
    # return the series as a xr.DataArray
    return q, reaches      


def reof(stack: xr.DataArray, n_modes: int=4):
    """
    Perform Rotated Empirical Orthogonal Function (REOF) on multi-temporal satellite images

    args:
        stack (xr.DataArray): DataArray of spatial temporal values with coord order of (t,y,x)
        n_modes (int): number of eof modes to use. default = 4

    returns:
        reof_ds (xr.Dataset): Dataset of REOF results with spatial modes, temporal modes, mean of data, and
                    explained variances as variables
    """

    from sklearn.decomposition import PCA


    def _ortho_rotation(components, method="varimax", tol=1e-6, max_iter=100):
        """Return rotated components."""
        nrow, ncol = components.shape
        rotation_matrix = np.eye(ncol)
        var = 0

        for _ in range(max_iter):
            comp_rot = np.dot(components, rotation_matrix)
            if method == "varimax":
                tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / nrow)
            elif method == "quartimax":
                tmp = 0
            u, s, v = np.linalg.svd(np.dot(components.T, comp_rot**3 - tmp))
            rotation_matrix = np.dot(u, v)
            var_new = np.sum(s)
            if var != 0 and var_new < var * (1 + tol):
                break
            var = var_new

        return np.dot(components, rotation_matrix)


    time_size = stack.sizes['time']
    lat_size = stack.sizes['lat']
    lon_size = stack.sizes['lon']
    space_size = lat_size*lon_size

    temporal_mean = np.nanmean(stack.values, axis=0)

    flat_da = stack.values.reshape((time_size, space_size))
    aoi_mask_flat = ~(np.isnan(flat_da).all(axis=0))
    flat_da = flat_da[:, aoi_mask_flat]
    flat_da = flat_da - flat_da.mean(axis=0, keepdims=True)

    # ----- Record the original index of each pixels inside the AOI to recover the flattened data back to geographical dimension -----
    flat2geo = np.arange(np.prod(space_size))[aoi_mask_flat].reshape(1,-1)

    # ----- sklearn PCA -----
    pca = PCA(n_components=n_modes)
    pcs = pca.fit_transform(flat_da)
    pcs = pcs.T
    S = pca.explained_variance_
    expvar  = pca.explained_variance_ratio_
    eofs = (pca.components_).T

    rotated_eofs = _ortho_rotation(eofs)

    # project the original time series data on the rotated eofs
    rotated_pcs = np.dot(flat_da, rotated_eofs)

    # get variance of each rotated mode
    rotated_var = np.var(rotated_pcs, axis=0)

    # get cumulative variance of all rotated modes
    total_rotated_var = rotated_var.cumsum()[-1]

    # get variance fraction of each rotated mode
    rotated_var_frac = ((rotated_var/total_rotated_var)*np.sum(expvar))

    # ----- Recover the flattened, AOI-only EOF array back to the geographical dimension -----
    rotated_eofs = rotated_eofs.T
    rotated_pcs = rotated_pcs.T

    sort_rotated_var_frac = np.argsort(-1*rotated_var_frac) # Index of explained variance in the descending order
    rotated_var_frac = rotated_var_frac[sort_rotated_var_frac]

    # sort modes based on variance fraction of REOF
    indx_rotated_var_frac_sort = np.expand_dims(sort_rotated_var_frac.data, axis=-1) # Mode X 1
    rotated_pcs = np.take_along_axis(rotated_pcs,indx_rotated_var_frac_sort,axis=0)

    rotated_eofs = np.take_along_axis(rotated_eofs,indx_rotated_var_frac_sort,axis=0)
    fill_eofs = np.ones((n_modes, space_size))*np.nan
    for ct_c in np.arange((rotated_eofs.shape[1])):
        fill_eofs[:,flat2geo[0,ct_c]] = rotated_eofs[:,ct_c]
    rec_eofs_img = fill_eofs.reshape(n_modes, lat_size, lon_size)

    # define data with variable attributes
    data_vars = {'spatial_modes':(['mode','lat','lon'], rec_eofs_img),
                 'temporal_modes':(['mode','time'], rotated_pcs),
                 'temporal_mean':(['lat','lon'], temporal_mean),
                 'explained_var':(['mode'], rotated_var_frac)
                }

    # define coordinates
    coords = {'time': (['time'], stack.time.values),
              'lat': (['lat'], stack.lat.values,{'units':'degrees North'}),
              'lon': (['lon'], stack.lon.values,{'units':'degrees East'}),
              'mode': (['mode'], np.arange(n_modes)+1)
             }


    # create dataset
    reof_ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
    )

    return reof_ds



def get_streamflow(lat: float, lon: float) -> Tuple[xr.DataArray, int]:
    """Function to get histroical streamflow data from the GeoGLOWS server
    based on geographic coordinates

    args:
        lat (float): latitude value where to get streamflow data
        lon (float): longitude value where to get streamflow data

    returns:
        xr.DataArray: DataArray object of streamflow with datetime coordinates
    """
    # ??? pass lat lon or do it by basin ???
    reach = streamflow.latlon_to_reach(lat,lon)
    # send request for the streamflow data
    q = streamflow.historic_simulation(reach['reach_id'])

    # rename column name to something not as verbose as 'streamflow_m^3/s'
    q.columns = ["discharge"]

    # rename index and drop the timezone value
    q.index.name = "time"
    q.index = q.index.tz_localize(None)

    # return the series as a xr.DataArray
    return q.discharge.to_xarray(), reach['reach_id']


def match_dates(original: xr.DataArray, matching: xr.DataArray) -> xr.DataArray:
    """Helper function to filter a DataArray from that match the data values of another.
    Expects that each xarray object has a dimesion named 'time'

    args:
        original (xr.DataArray): original DataArray with time dimension to select from
        matching (xr.DataArray): DataArray with time dimension to compare against

    returns:
        xr.DataArray: DataArray with values that have been temporally matched
    """

    # return the DataArray with only rows that match dates
    return original.where(original.time.isin(matching.time),drop=True)

def fits_to_files(fit_dict: dict,out_dir: str):
    """Procedure to save coeffient arrays stored in dictionary output from `find_fits()` to npy files

    args:
        fit_dict (dict): output from function `find_fits()`
        out_dir (str): directory to save coeffients to
    """

    out_dir = Path(out_dir)

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    for k,v in fit_dict.items():
        if k.endswith("coeffs"):
            components = k.split("_")
            name_stem = f"poly_{k.replace('_coeffs','')}"
            coeff_file = out_dir / f"{name_stem}.npy"
            np.save(str(coeff_file), v)

    return

def find_fits(reof_ds: xr.Dataset, q_df: xr.DataArray, stack: xr.DataArray, train_size: float = 0.7, random_state: int = 0, ):
    """Function to fit multiple polynomial curves on different temporal modes and test results

    """        
    
    X = q_df
    y = reof_ds.temporal_modes

    # ---- Randomly split data into 2 groups -----
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

    logger.debug(X_train)
    logger.debug(X_test)

    spatial_test = stack.where(stack.time.isin(y_test.time),drop=True)

    shape3d = spatial_test.shape
    spatial_shape = shape3d[1:]
    shape2d = (shape3d[0],np.prod(spatial_shape))

    spatial_test_flat = xr.DataArray(
        spatial_test.values.reshape(shape2d),
        coords = [spatial_test.time,np.arange(shape2d[1])],
        dims=['time','space']
    )

    non_masked_idx= np.where(np.logical_not(np.isnan(spatial_test_flat[0,:])))[0]

    modes = reof_ds.mode.values

    fit_dict = dict()
    dict_keys = ['fit_r2','pred_r','pred_rmse']

    for mode in modes:

        logger.debug(mode)
        y_train_mode = y_train.sel(mode=mode)
        y_test_mode = y_test.sel(mode=mode)

        for order in range(1,4):

            # apply polynomial fitting
            c = np.polyfit(X_train,y_train_mode,deg=order)
            f = np.poly1d(c)

            y_pred = f(X_test)

            synth_test = synthesize(reof_ds,X_test,f,mode=mode)                      
             

            synth_test_flat = xr.DataArray(
                synth_test.values.reshape(shape2d),
                coords = [synth_test.time,np.arange(shape2d[1])],
                dims=['time','space']
            )

            # calculate statistics
            # calculate the stats of fitting on a test subsample
            temporal_r2 = metrics.r2_score(y_test_mode, y_pred)

            temporal_r = -999 if temporal_r2 < 0 else np.sqrt(temporal_r2)

            # check the synthesis stats comapared to observed data
            space_r2 = metrics.r2_score(
                spatial_test_flat[:,non_masked_idx],
                synth_test_flat[:,non_masked_idx],
            )
            space_r= -999 if space_r2 < 0 else np.sqrt(space_r2)

            space_rmse = metrics.mean_squared_error(
                spatial_test_flat[:,non_masked_idx],
                synth_test_flat[:,non_masked_idx],
                squared=False
            )

            # pack the resulting statistics in dictionary for the loop
            #stats = [temporal_r2,space_r,space_rmse]
            stats = [temporal_r2, temporal_r, space_rmse]
            loop_dict = {f"mode{mode}_order{order}_{k}":stats[i] for i,k in enumerate(dict_keys)}
            loop_dict[f"mode{mode}_order{order}_coeffs"] = c
            logging.debug(loop_dict)
            # merge the loop dictionary with the larger one
            fit_dict = {**fit_dict,**loop_dict}

    return fit_dict


def sel_best_fit(fit_dict: dict, metric: str = "r",ranking: str = "max") -> tuple:
    """Function to extract out the best fit based on user defined metric and ranking

    args:
        fit_dict (dict): output from function `find_fits()`
        metric (str, optional): statitical metric to rank, options are 'r', 'r2' or 'rmse'. default = 'r'
        ranking (str, optional): type of ranking to perform, options are 'min' or 'max'. default = max

    returns:
        tuple: tuple of values containg the coefficient key, mode, and coefficients of best fit
    """
    def max_ranking(old,new):
        key,ranker = old
        k,v = new
        if v > ranker:
            key = k
            ranker = v
        return (key,ranker)
    
    def min_ranking(old,new):
        key,ranker = old
        k,v = new
        if v < ranker:
            key = k
            ranker = v
        return (key,ranker)

    if metric not in ["r","r2","rmse"]:
        raise ValueError("could not determine metric to rank, options are 'r', 'r2' or 'rmse'")

    if ranking == "max":
        ranker = -1
        ranking_f = max_ranking
    elif rankin == "min":
        ranker = 999
        ranking_f = min_ranking
    else:
        raise ValueError("could not determine ranking, options are 'min' or 'max'")

    ranked = ("",ranker)
    
    for k,v in fit_dict.items():
        if k.endswith(metric):
            ranked = ranking_f(ranked,(k,v))

    key_split = ranked[0].split("_")
    ranked_key = "_".join(key_split[:-2] + ["coeffs"])
    coeffs = fit_dict[ranked_key]
    mode = int(key_split[0].replace("mode",""))

    return (ranked_key,mode,coeffs)

def synthesize_indep(reof_ds: xr.Dataset, q_df: xr.DataArray, model_mode_order, model_path='.\\model_path\\'):
    """Function to synthesize data at time of interest and output as DataArray
   
    """
    mode_list=list(model_mode_order)
    for num_mode in mode_list:
        #for order in range(1,4):
             
        f = np.poly1d(np.load(model_path+'\poly'+'{num:0>2}'.format(num=str(num_mode))+'_deg'+'{num:0>2}'.format(num=model_mode_order[str(num_mode)])+'.npy'))
        #logger.debug('{model_mode_order[str(mode)]:0>2}'+'.npy')
        
        
        y_vals = xr.apply_ufunc(f, q_df)
        logger.debug(y_vals)
        
        synth = y_vals * reof_ds.spatial_modes.sel(mode=int(num_mode)) # + reof_ds.center
        
        synth = synth.astype(np.float32).drop("mode").sortby("time")

        return synth
             

def synthesize(reof_ds: xr.Dataset, q_df: xr.DataArray, polynomial: np.poly1d, mode: int = 1) -> xr.DataArray:
    """Function to synthesize data from reof data and regression coefficients.
    This will also format the result as a geospatially aware xr.DataArray

    args:
        reof_ds (xr.Dataset):
        q_df (xr.DataArray):
        polynomial (np.poly1d):
        mode (int, optional):


    returns:
        xr.DataArray: resulting synthesized data based on reof temporal regression
            and spatial modes
    """

    y_vals = xr.apply_ufunc(polynomial,q_df)
    
    logger.debug(y_vals)

    synth = y_vals * reof_ds.spatial_modes.sel(mode=mode) + reof_ds.center

    # drop the unneeded mode dimension
    # force sorting by time in case the array is not already
    synth = synth.astype(np.float32).drop("mode").sortby("time")
    

    return synth
