import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import *
import numpy as np
from scipy.ndimage import maximum_filter, correlate, correlate1d
from skimage.morphology import extrema
from scipy import ndimage as ndi
import time
import hashlib
np.set_printoptions(precision=2,suppress=True,edgeitems=10, linewidth=100000)



FILENAME = ""
FOLDER = ".\\data\\raw\\"
BETAS = [0,-67/180*np.pi] #phase calibration
R0 = 2.5 # range calibration
LOG = True
N = 256
M = 256

file = np.load(FOLDER + FILENAME + '.npz', allow_pickle=True)
lst = file.files

print('FILE : ', FILENAME)
for item in lst:
    print(item)
    print("shape of " + item + " : ", np.shape(file[item]))
data = file['data']
data_times = file['data_times']
chirp = file['chirp']
bg_data = file['background']

f0 = chirp[0]
B = chirp[1]
Ms = int(chirp[2])
Mc = int(chirp[3])
Ts = chirp[4]
Tc = chirp[5]
frames = len(data_times)
c = 299_792_458
T = Ms*Ts
wlength = c/f0
k = 2*np.pi/wlength
L = 0.0625
print('Wavelength : ',wlength)
pause = int(len(data[0,0,:])/Mc) - Ms
M_avec_pause = int(len(data[0,0,:])/Mc)

print("Chirp duration : ", T)
print("Chirp repetition period : ",Tc)
print("Sample period : ", Ts)
print("Number of frames : ",frames)
print("Number of samples per frame : ",len(data[0,0,:]))
print("Number of chirp per frame:", Mc)
print("Number of samples per chirp:", Ms)
print("Number of samples per chirp + pause:", M_avec_pause)
inter = np.diff(data_times).mean()*1000
print('Average frame interval : ',inter)

mean_bg = np.mean(bg_data, axis=0)
print('shape of mean_bg', np.shape(mean_bg))
def remove_background(data): 
    for i in range(frames):
        data[i] = data[i] - mean_bg
    return data
#data = remove_background(data)


dmax = c*Ms/(2*B)
d = np.linspace(0,dmax,N)
d_ = np.linspace(0,dmax,Ms)
print('Max distance : ', dmax)
print('Distance resolution : ', dmax/Ms)
vmax = 1/4*c/(f0*Tc)
v = np.linspace(-vmax,vmax,N)
v_ = np.linspace(-vmax,vmax,Mc)
print('Speed resolution : ', 2*vmax/Mc)


def remove_pauses(data):
    newdata = np.zeros((frames,4,Mc,Ms),dtype=complex)
    for f in range(frames):
        for m in range(4):
            for ns in range(int(Ms)):
                for nc in range(int(Mc)):
                    newdata[f, m, nc, ns] = data[f, m, nc*M_avec_pause + ns]
    return newdata
data = remove_pauses(data)


antenna1 = np.zeros((frames,Mc,Ms),dtype=complex)
for f in range(frames):
    for ns in range(int(Ms)):
        for nc in range(int(Mc)):
            antenna1[f,nc,ns] = data[f, 0, nc, ns] + 1j*data[f, 1, nc, ns]


antenna2 = np.zeros((frames,Mc,Ms),dtype=complex)
for f in range(frames):
    for ns in range(int(Ms)):
        for nc in range(int(Mc)):
            antenna2[f,nc,ns] = data[f, 2, nc, ns] + 1j*data[f, 3, nc, ns]


    
#Removing null velocities
def remove_0freq(ant): 
    for i in range(Ms):
        offset = 0
        for j in range(Mc):
            offset += ant[:,j,i]
        offset = offset/Mc
        for j in range(Mc):
            ant[:,j,i] -= offset
    return ant
antenna1 = remove_0freq(antenna1)
antenna2 = remove_0freq(antenna2)




















def hash_arguments(*args,**kwargs):
    key = ''
    for arg in args:
        if type(arg) is not np.ndarray:
            key += str(arg)
        else:
            key += hashlib.sha1(arg.tobytes()).hexdigest()
            key += str(arg.shape)
            key += str(arg.strides)
    for arg in kwargs.values():
        if type(arg) is not np.ndarray:
            key += str(arg)
        else:
            key += hashlib.sha1(arg.tobytes()).hexdigest()
            key += str(arg.shape)
            key += str(arg.strides)
    return key

class mycache:
    def __init__(self, function):
        self.cache = {}
        self.function = function

    def __call__(self, *args, **kwargs):        
        key = hash_arguments(*args)
        if key in self.cache:
            return self.cache[key]

        value = self.function(*args, **kwargs)
        if type(value) == np.ndarray:
            cache = value.copy()
            self.cache[key] = cache
        elif type(value) == tuple:
            cache = []
            for val in value:
                if type(val) == np.ndarray:
                    cache.append(val.copy())
                else:
                    cache.append(val)
            cache = tuple(cache)
            self.cache[key] = cache
        else:
            self.cache[key] = value
        return value



def to_shape(a, shape):
    """https://stackoverflow.com/questions/56357039/numpy-zero-padding-to-match-a-certain-shape"""
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')


def mysinc(x,y,ms,mc,n):
    sol = np.zeros_like(x)
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            if x[i,j] == 0 and y[i,j] == 0:
                sol[i,j] = 1
            elif x[i,j]==0:
                sol[i,j] = np.sin(ms/n*np.pi*y[i,j])/np.sin(np.pi/n*y[i,j])/ms
            elif y[i,j]==0:
                sol[i,j] = np.sin(mc/n*np.pi*x[i,j])/np.sin(np.pi/n*x[i,j])/mc
            else:
                sol[i,j] = np.sin(mc/n*np.pi*x[i,j])/np.sin(np.pi/n*x[i,j])/mc * np.sin(ms/n*np.pi*y[i,j])/np.sin(np.pi/n*y[i,j])/ms
    return sol


def mysinc1d(x,ms,mc,n):
    sol1 = np.zeros_like(x)
    sol2 = np.zeros_like(x)
    for i in range(np.shape(x)[0]):
        if x[i] == 0:
            sol1[i] = 1
            sol2[i] = 1
        else:
            sol1[i] = np.sin(ms/n*np.pi*x[i])/np.sin(np.pi/n*x[i])/ms
            sol2[i] = np.sin(mc/n*np.pi*x[i])/np.sin(np.pi/n*x[i])/mc
    return np.array((sol1, sol2))


def gkern(l, sig):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    val = kernel / np.sum(kernel)
    return val


def disk_footprint(size):
    a, b = size//2, size//2
    n = size
    r = size
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    array = np.zeros((n, n))
    array[mask] = 1
    return array

def check_indices_in_lims(indices, xlims, ylims):
    for index in indices:
        if index[0] < xlims[0] or index[0] > xlims[1]\
            or index[1] < ylims[0] or index[1] > ylims[1]:
            return False
    return True

# unidentified BUG when caching this

def sorted_indx(arr, n, xlims, ylims):
    indices = np.ones((n,2))*-100000
    flat = arr.ravel()
    search = n*3
    write = 0
    while write < n and search < len(flat):
        all_indx = np.argpartition(flat, -search)[-search:]
        all_indx = np.transpose(np.unravel_index(all_indx[np.argsort(-flat[all_indx])], np.shape(arr)))
        write = 0
        for i in range(len(all_indx)):
            idx = all_indx[i]
            if idx[0] > xlims[0] and idx[0] < xlims[1] and idx[1] > ylims[0] and idx[1] < ylims[1]:
                indices[write] = idx
                write+=1
            if(write >= n):break
        search = search*2
    if search >= len(flat):
        print('WARNING not enough maxima found : decreasing requierements')
        return None
    return indices


def hmax(arr, h, nidx, xlims, ylims):
    indx = None
    it = 0
    while indx is None and it < 1000:
        h_maxima = extrema.h_maxima(arr, h)
        mask = h_maxima != 0
        maxs = np.where(mask,arr,0) 
        indx = sorted_indx(maxs, nidx, xlims, ylims)
        h = h/2
        it+=1
    if it == 1000: print('WARNING not enough maxima found in limits')
    return indx, maxs


def max_filter(arr, footprint, nidx, xlims, ylims):
    indx = None
    it = 0
    while indx is None and it < 1000 and footprint.shape[0] > 2:
        C_max = maximum_filter(arr, footprint=footprint, mode='wrap')
        C_maxima = (arr == C_max)
        C_max[~C_maxima] = 0
        indx = sorted_indx(C_max, nidx, xlims, ylims)
        footprint = disk_footprint(footprint.shape[0] - 1)
        it+=1
    if it == 1000 or footprint.shape[0] == 2: print('WARNING not enough maxima found in limits')
    return indx, C_max

def compute_corrfilter(size, corrfiltertype, Ms, Mc, N):
    ratio = size/N
    x = np.linspace(-size/2,size/2,size)
    y = np.linspace(-size/2,size/2,size)
    X_fltr,Y_fltr = np.meshgrid(x,y)

    if corrfiltertype == 'sinc1d':
        fltr = np.abs(mysinc1d(x,Ms,Mc,N)) #Warning : M not implemented
    
    elif corrfiltertype == 'sinc(hann)':
        start = N//2-(size//2)   
        wx = np.hanning(Ms)
        fftx = np.fft.fftshift(np.abs(np.fft.fft(wx, N)))[start:start+size]
        wy = np.hanning(Mc)
        ffty = np.fft.fftshift(np.abs(np.fft.fft(wy, N)))[start:start+size]
        fltr = np.array([fftx, ffty])
    
    elif corrfiltertype == 'sinc':
        fltr = np.abs(mysinc(X_fltr,Y_fltr,Ms,Mc,N)) #Warning : M not implemented
    
    elif corrfiltertype == 'hsinc':   
        fltr = np.abs(mysinc(X_fltr,Y_fltr,Ms,Mc,N)) #Warning : M not implemented
        width = (N/Ms)//2
        for i in range(size):
            if i not in np.arange(size//2-width,size//2+width+1):
                fltr[i]*=0
    
    elif corrfiltertype == 'hann':
        fltr = np.hanning(size)[:,None]*np.hanning(size)[None,:]
    
    elif corrfiltertype == 'gauss':
        fltr = gkern(size,size/3)*0.4
    
    else:
        fltr = np.zeros((size,size))
        fltr[size//2, size//2] = 1
    return fltr

def global_nearest_neighbor(previous_positions, current_positions, threshold):
    """
    Perform global nearest neighbor association between previous and current positions.

    Args:
        previous_positions: N x 2 array of previous positions.
        current_positions: M x 2 array of current positions.
        threshold: Maximum distance between previous and current positions for association.

    Returns:
        associations: List of length M with the index of the associated previous position (-1 for no association).
    """
    num_previous = previous_positions.shape[0]
    num_current = current_positions.shape[0]
    cost_matrix = np.zeros((num_previous, num_current))

    # Compute pairwise Euclidean distances between previous and current positions
    for i in range(num_previous):
        for j in range(num_current):
            cost_matrix[i, j] = np.linalg.norm(previous_positions[i] - current_positions[j])

    associations = np.full(num_current, -1, dtype=int)  # Initialize all associations to -1

    # Greedy matching: find the closest unassigned current position for each previous position
    for i in range(num_previous):
        j = np.argmin(cost_matrix[i, :])
        if cost_matrix[i, j] <= threshold:
            if associations[j] == -1:
                associations[j] = i
            else:
                # If multiple current positions are assigned to the same previous position,
                # keep the one with the closest distance
                current_distance = cost_matrix[i, j]
                previous_distance = np.linalg.norm(previous_positions[associations[j]] - current_positions[j])
                if current_distance < previous_distance:
                    associations[j] = i

    return associations.tolist()

def print_target_infos(target_infos, ntgt):
        print('         Target :',end='\t')
        for i in range(ntgt): 
            print(f'{i}', end='\t')
        print('\nEstimated range :',end='\t')
        for i in range(ntgt): 
            print(f'{target_infos[i][0]:0,.2f}', end='\t')
        print('\nEstimated speed :',end='\t')
        for i in range(ntgt): 
            print(f'{target_infos[i][1]:0,.2f}', end='\t')
        print('\nEstimated angle :',end='\t')
        for i in range(ntgt): 
            print(f'{target_infos[i][2]:0,.2f}', end='\t')
        print('\nPhase difference:',end='\t')
        for i in range(ntgt): 
            print(f'{target_infos[i][3]:0,.2f}')









MAXTARGETS = 6
def init(frames, N, Ms, Mc, v, d, R0, vmax, dmax, antenna1, antenna2, k, L, BETAS):

    zeros = np.zeros((frames,N,N))
    VERBOSE = True
    
    ANTENNA = 0
    FRM = 0
    NTARGETS = 1

    WINDOWED =      [np.zeros((frames,N,N), np.complex128),np.zeros((frames,N,N), np.complex128)]
    FFTS =          [np.zeros((frames,N,N), np.complex128),np.zeros((frames,N,N), np.complex128)]
    C =             [np.zeros((frames,N,N)),np.zeros((frames,N,N))]
    PREFILTERED =   [np.zeros((frames,N,N)),np.zeros((frames,N,N))]
    FILTERED =      [np.zeros((frames,N,N)),np.zeros((frames,N,N))]
    MAXFILTERED =   [zeros.copy(),zeros.copy()]
    MAXINDX =       [np.zeros((frames, MAXTARGETS, 2),int), np.zeros((frames, MAXTARGETS, 2),int)]
    
    PF_UPDATED =    np.zeros(frames, bool)
    F_UPDATED =     np.zeros(frames, bool)
    MAX_UPDATED =   np.zeros(frames, bool)
    
    FOOTPRINT = compute_corrfilter(3, 'none', Ms, Mc, N)
    CORRFLTR = compute_corrfilter(3, 'none', Ms, Mc, N)
    NOISEFLTR = compute_corrfilter(3, 'none', Ms, Mc, N)

    EMPTY_POS = np.zeros((MAXTARGETS, frames,2))
    EMPTY_POS[:,:,1] = -5
    POSITIONS = EMPTY_POS.copy()   
    
    frame_arr = np.arange(frames)
    EMPTY_ANGLES = np.zeros((MAXTARGETS, frames, 2))
    EMPTY_ANGLES[:,:,0] = frame_arr
    EMPTY_ANGLES[:,:,1] = -5
    ANGLES = EMPTY_ANGLES.copy()#WARNING in degrees so plots are less laggy
    SM_ANGLES = ANGLES.copy()

    EMPTY_RANGES = np.zeros((MAXTARGETS, frames, 2))
    EMPTY_RANGES[:,:,0] = frame_arr
    EMPTY_RANGES[:,:,1] = 1000
    RANGES = EMPTY_RANGES.copy()
    SM_RANGES = RANGES.copy()

    

    x = np.linspace(-vmax,vmax,N)
    y = np.linspace(0,dmax,N)
    X,Y = np.meshgrid(x,y)

    phi = np.linspace(0,np.pi,360+1)
    uy = np.cos(phi)
    uz = np.sin(phi)
    u_hat = np.array([uy,uz]).T

    def reset(pf,f,max):
        nonlocal POSITIONS, ANGLES, RANGES, SM_ANGLES, SM_RANGES
        if VERBOSE: print(f"reseting : pf:{pf}\tf:{f}\tmax:{max}\t")
        nonlocal PF_UPDATED, F_UPDATED, MAX_UPDATED, POSITIONS
        if pf: PF_UPDATED[:] =    False
        if f: F_UPDATED =     np.zeros(frames, bool)
        if max: MAX_UPDATED =   np.zeros(frames, bool)
        POSITIONS = EMPTY_POS.copy()
        ANGLES = EMPTY_ANGLES.copy()
        RANGES = EMPTY_RANGES.copy()
        SM_RANGES = RANGES.copy()
        SM_ANGLES = ANGLES.copy() 
        axes2[1,0].clear()
        #canvas2.draw_idle()
    

    def clear_pos(val=None):
        nonlocal POSITIONS, ANGLES, RANGES, SM_ANGLES, SM_RANGES
        POSITIONS = EMPTY_POS.copy()
        ANGLES = EMPTY_ANGLES.copy()
        RANGES = EMPTY_RANGES.copy()
        SM_RANGES = RANGES.copy()
        SM_ANGLES = ANGLES.copy()
        axes2[1,0].clear()
        update_scatterplot()
        
    def update_tracks(val=None):
        nonlocal SM_ANGLES, SM_RANGES
        axes2[1,0].clear()
        l = slider_smoothing_param.get()
        for t in range(MAXTARGETS):
            flim = (int(entry_FRL.get())-1, int(entry_FRH.get()))
            SM_ANGLES[t,flim[0]:flim[1],1] = ndi.uniform_filter1d(ANGLES[t,flim[0]:flim[1],1], l)
            SM_RANGES[t,flim[0]:flim[1],1] = ndi.uniform_filter1d(RANGES[t,flim[0]:flim[1],1], l)
            fx = np.cos(SM_ANGLES[t,:,1]/180*np.pi)*SM_RANGES[t,:,1]
            fy = np.sin(SM_ANGLES[t,:,1]/180*np.pi)*SM_RANGES[t,:,1]
            if t < NTARGETS:
                if var_tracking.get() == 'raw smooth':
                    axes2[1,0].plot(fx[flim[0]:flim[1]],fy[flim[0]:flim[1]], c=cols[t])
                elif var_tracking.get() == 'GNN':
                    pass
                elif var_tracking.get() == 'bsplines':
                    pass
        axes2[1,0].set_ylim((-5,dmax))
        axes2[1,0].set_xlim((-dmax,dmax))
        update_scatterplot()
        canvas2.draw_idle()

    def compute_all_frames(val=None):
        from datetime import datetime
        global chirp
        slider_frame.set(0)
        update_frame()
        reset(True,True,True)
        flim = (int(entry_FRL.get()), int(entry_FRH.get())+1)
        for f in range(flim[0], flim[1]):
            slider_frame.set(f)
            update_frame()
        if var_savepos.get() == 1:
            now = datetime.now()
            np.savez(".\\data\\processed\\"+FILENAME+"_posdata"+now.strftime("%d-%m-%Y_%H%M%S"),
                positions = POSITIONS[flim[0]: flim[1]],
                ranges = RANGES[flim[0]: flim[1]],
                angles= ANGLES[flim[0]: flim[1]],
                chirp = chirp,
                window = var_window.get(),
                noisefilter = var_noisefilter.get(),
                noisefiltersize = slider_noisefiltersize.get(),
                corrfilter = var_corrfilter.get(),
                corrfiltersize = slider_corrfiltersize.get(),
                maxmethod = var_maxmethod.get(),
                maxmethodsize = slider_maxmethodsize.get(),
                maxmethodparam = slider_maxmethodparameter.get(),
            )
        update_tracks()
        
    
    def update_window(val=None):
        nonlocal WINDOWED
        #windowing
        if var_window.get() == 'hann2d':
            win = np.hanning(Mc)[:,None]*np.hanning(Ms)[None,:]
        elif var_window.get() == 'blackman2d':
            win = np.blackman(Mc)[:,None]*np.blackman(Ms)[None,:]
        else:
            win = np.ones((Mc,Ms))

        WINDOWED[0] = win * antenna1
        WINDOWED[1] = win * antenna2
        update_ffts()

    def update_ffts():
        nonlocal FFTS, WINDOWED
        fft1 = np.transpose(np.fft.fftshift(np.fft.ifft2(WINDOWED[0],(N,N)), axes=(1)), (0,2,1))
        fft2 = np.transpose(np.fft.fftshift(np.fft.ifft2(WINDOWED[1],(N,N)), axes=(1)), (0,2,1))

        FFTS[0] = fft1*np.exp(-1j*(BETAS[0]))
        FFTS[1] = fft2*np.exp(-1j*(BETAS[1]))

        C[0] = (np.abs(FFTS[0]))**2
        C[1] = (np.abs(FFTS[1]))**2

        for m in range(2):
            for f in range(frames):
                for j in range(N):
                    if C[m][f,j,N//2] < 1e-5:
                        C[m][f,j,N//2] = 0
                    if C[m][f,j,N//2] < 1e-5:
                        C[m][f,j,N//2] = 0
        reset(True,True,True)
        update_frame()

        

    def update_frame(val=None):
        nonlocal FRM, ANTENNA
        slider_frame.configure(from_=int(entry_FRL.get()),to=int(entry_FRH.get()))
        print(f'\nframe {FRM+1}/{frames}')
        FRM = slider_frame.get()-1
        title = var_antenna.get()
        ANTENNA = int(title[-1])-1
        update_noisefilter()

    def update_noisefilter(val=None):
        if val is not None: reset(True, True, True)
        nonlocal FRM, PREFILTERED, FILTERED, MAXFILTERED, MAXINDX, CORRFLTR, NOISEFLTR
        size = slider_noisefiltersize.get()
        noisefiltertype = var_noisefilter.get()
        if not PF_UPDATED[FRM]:
            if VERBOSE: 
                t = time.time()
                print("updating noisefilter", end='\t')
            for m in range(2):
                if noisefiltertype == 'gauss2d':
                    NOISEFLTR = gkern(4*size+1, size)
                    PREFILTERED[m][FRM] = ndi.gaussian_filter(C[m][FRM], size)
                else:
                    fltr = np.zeros((size,size))
                    fltr[size//2, size//2] = 1  
                    NOISEFLTR = fltr
                    PREFILTERED[m][FRM] = C[m][FRM]
            PF_UPDATED[FRM] = True
            if VERBOSE: 
                print("time taken :", time.time()-t)
        if var_filterplot.get() == 'noise':
            update_filterplot()
        update_corrfilter()

    def update_corrfilter(val=None):
        if val is not None: reset(False, True, True)
        nonlocal FRM, PREFILTERED, FILTERED, CORRFLTR, F_UPDATED
        if not F_UPDATED[FRM]:
            if VERBOSE: 
                print("updating correlation filter", end='\t')
                t = time.time()
            size = slider_corrfiltersize.get()
            corrfiltertype = var_corrfilter.get()
            CORRFLTR =  compute_corrfilter(size, corrfiltertype, Ms,Mc, N)
            
            for m in range(2):
                if np.shape(CORRFLTR)[0] == 2:
                    FILTERED[m][FRM] = correlate1d(PREFILTERED[m][FRM],CORRFLTR[0], 0, mode='wrap')
                    FILTERED[m][FRM] = correlate1d(FILTERED[m][FRM],CORRFLTR[1], 1, mode='wrap')
                else:
                    FILTERED[m][FRM] = correlate(PREFILTERED[m][FRM],CORRFLTR,mode='wrap')

            F_UPDATED[FRM] = True
            if VERBOSE: 
                print("time taken :", time.time()-t)
        if var_filterplot.get() == 'corr':
            update_filterplot()

        update_maxfilter()


    def update_maxfilter(val=None):
        if val is not None: reset(False, False, True)
        nonlocal FRM, FILTERED, MAXFILTERED, MAXINDX, FOOTPRINT
        if not MAX_UPDATED[FRM]:
            if VERBOSE: 
                print("updating max filter", end='\t')
                t = time.time()
            rlim=(float(entry_RRL.get()), float(entry_RRH.get()))
            vlim=(float(entry_VRL.get()), float(entry_VRH.get()))
            rlimidx = (np.searchsorted(d-R0, rlim[0], side='right'), np.searchsorted(d-R0, rlim[1], side='right'))
            vlimidx = (np.searchsorted(v, vlim[0], side='right'), np.searchsorted(v, vlim[1], side='right'))
            for m in range(2):
                tomaximize = FILTERED[m][FRM].copy()
                if var_maxmethod.get() == 'maxfilter':
                    FOOTPRINT = disk_footprint(slider_maxmethodsize.get())
                    MAXINDX[m][FRM], MAXFILTERED[m][FRM] = max_filter(tomaximize, FOOTPRINT, \
                                                                        MAXTARGETS, rlimidx, vlimidx)
                elif var_maxmethod.get() == 'h-max':
                    MAXINDX[m][FRM], MAXFILTERED[m][FRM] = hmax(tomaximize,slider_maxmethodparameter.get(), \
                                                                MAXTARGETS, rlimidx, vlimidx)
                print('test')
                print(MAXINDX[m][FRM])
            MAX_UPDATED[FRM] = True
            if VERBOSE: 
                print("time taken :", time.time()-t)
        if var_filterplot.get() == 'maxfootprint':
            update_filterplot()
        
        update_plot1()
        update_plot2()
        update_scatterplot()


    def update_filterplot(val=None):
        nonlocal CORRFLTR, NOISEFLTR
        title = var_filterplot.get()
        if title == 'corr':
            fltr = CORRFLTR
            if np.shape(CORRFLTR)[0] == 2:
                fltr = CORRFLTR[0][:,None] * CORRFLTR[1][None,:]
        elif title == 'noise':
            fltr = NOISEFLTR
        elif title == 'maxfootprint':
            fltr = FOOTPRINT

        #pad_width_fltr = ((np.shape(X)[0] - np.shape(fltr)[0])//2, (np.shape(X)[1] - np.shape(fltr)[1])//2)
        axes1[1,0].set_title(title + " filter plot")
        pcm3.set_array(to_shape(fltr, (N,N)))
        vmin, vmax = (fltr.min()), fltr.max()
        pcm3.set_clim(vmin, vmax)
        pcm3.norm.autoscale(pcm3._A)
        pcm3.colorbar.update_normal(pcm3.colorbar.mappable)
        canvas1.draw_idle()


    def update_scatterplot(val=None):
        nonlocal FFTS, FRM, NTARGETS, MAXINDX, NTARGETS, POSITIONS, ANGLES, RANGES, SM_ANGLES, SM_RANGES
        NTARGETS = slider_targets.get()
        angle_est_type = var_angle_est.get()
        target_infos = []
        


        for t in range(MAXTARGETS):
            R = (d[MAXINDX[0][FRM][t][0]] + d[MAXINDX[1][FRM][t][0]])/2 -R0
            V = (v[MAXINDX[0][FRM][t][1]] + v[MAXINDX[1][FRM][t][1]])/2
            dtheta = np.angle(FFTS[0][FRM][tuple(MAXINDX[0][FRM][t])]*np.conj(FFTS[1][FRM][tuple(MAXINDX[1][FRM][t])]))
            
            if angle_est_type == 'arg':
                bestphi = np.arccos((dtheta)/np.pi)
            elif angle_est_type == 'max':
                phased_resp_ = FFTS[0][FRM][tuple(MAXINDX[0][FRM][t])] + FFTS[1][FRM][tuple(MAXINDX[1][FRM][t])]*np.exp(1j*k*u_hat[:,0]*L)
                bestphi = phi[np.argmax(np.abs(phased_resp_))]
            POSITIONS[t][FRM] = np.array([np.cos(bestphi)*R, np.sin(bestphi)*R])
            ANGLES[t][FRM][1] = bestphi/np.pi*180
            RANGES[t][FRM][1] = R
            if(t < NTARGETS):
                targets_yz[t].set_offsets(np.array([np.cos(bestphi)*R, np.sin(bestphi)*R], ndmin=2))
                markers00[t].set_offsets(np.array([V, R], ndmin=2))
                markers01[t].set_offsets(np.array([V, R], ndmin=2))
                target_infos.append((R,V,bestphi/np.pi*180,dtheta/np.pi*180))
                old_targets_yz[t].set_offsets(POSITIONS[t])
                if var_estimplots.get() == 'raw':
                    angles[t].set_offsets(ANGLES[t])
                    ranges[t].set_offsets(RANGES[t])
                else:
                    angles[t].set_offsets(SM_ANGLES[t])
                    ranges[t].set_offsets(SM_RANGES[t])

            else:
                targets_yz[t].set_offsets(np.array([0,-5], ndmin=2))
                old_targets_yz[t].set_offsets(EMPTY_POS[t])
                angles[t].set_offsets(EMPTY_ANGLES[t])
                ranges[t].set_offsets(EMPTY_RANGES[t])
        if VERBOSE:print_target_infos(target_infos, NTARGETS)
        update_marker00()
        update_marker01()
        canvas1.draw_idle()
        canvas2.draw_idle()

    def update_marker00(val=None):
        if VERBOSE : print("updating plot1 target markers")
        if var_marker00.get() == 0:
            n = 0
        else:
            n= NTARGETS

        for t in range(MAXTARGETS):
            if t < n:
                R = d[MAXINDX[ANTENNA][FRM][t][0]]-R0
                V = v[MAXINDX[ANTENNA][FRM][t][1]]
                markers00[t].set_offsets(np.array([V, R], ndmin=2))
            else:
                markers00[t].set_offsets(np.array([0, -10], ndmin=2))
        canvas1.draw_idle()

    def update_marker01(val=None):
        if VERBOSE : print("updating plot2 target markers")
        if var_marker01.get() == 0:
            n = 0
        else:
            n= NTARGETS
        for t in range(MAXTARGETS):
            if t < n:
                R = d[MAXINDX[ANTENNA][FRM][t][0]]-R0
                V = v[MAXINDX[ANTENNA][FRM][t][1]]
                markers01[t].set_offsets(np.array([V, R], ndmin=2))
            else:
                markers01[t].set_offsets(np.array([0, -10], ndmin=2))
        canvas1.draw_idle()

    def update_plot1(val=None):
        toplot=0
        title = var_plot1.get()
        if title == "windowed":
            toplot = C[ANTENNA][FRM]
        elif title == "prefiltered":
            toplot = PREFILTERED[ANTENNA][FRM]
        elif title == "filtered":
            toplot = FILTERED[ANTENNA][FRM]
        elif title == "maxfiltered":
            toplot = MAXFILTERED[ANTENNA][FRM]
        else:
            toplot = C[ANTENNA][FRM]
        axes1[0,0].set_title("abs(ifft2)²")
        if var_logplot1.get() == 1:
            toplot = np.log10(toplot)
        pcm1.set_array(toplot)
        vmin, vmax = (toplot.min()), toplot.max()
        pcm1.set_clim(vmin, vmax)
        pcm1.norm.autoscale(pcm1._A)
        pcm1.colorbar.update_normal(pcm1.colorbar.mappable)
        canvas1.draw_idle()

    def update_plot2(val=None):
        toplot=0
        title = var_plot2.get() 
        if title == "windowed":
            toplot = C[ANTENNA][FRM]
        elif title == "prefiltered":
            toplot = PREFILTERED[ANTENNA][FRM]
        elif title == "filtered":
            toplot = FILTERED[ANTENNA][FRM]
        elif title == "maxfiltered":
            toplot = MAXFILTERED[ANTENNA][FRM]
        else:
            toplot = C[ANTENNA][FRM]
        axes1[0,1].set_title(title + " fft plot")
        if var_logplot2.get() == 1:
            toplot = np.log10(toplot)
        pcm2.set_array(toplot)
        vmin, vmax = (toplot.min()), toplot.max()
        pcm2.set_clim(vmin, vmax)
        pcm2.norm.autoscale(pcm2._A)
        pcm2.colorbar.update_normal(pcm2.colorbar.mappable)
        canvas1.draw_idle()
    

    # Créer la fenêtre principale avec un titre
    root = Tk()
    root.title("Paramètres")
    root.geometry("1220x480")
    Grid.rowconfigure(root, 0, weight=1)
    Grid.columnconfigure(root, 0, weight=1)
    # Create the canvas and add the plot to it
    
    window1 = Toplevel(root)
    window1.geometry("1220x720")
    window1.title("Plots")
    fig1, axes1 = plt.subplots(2, 2)
    canvas1 = FigureCanvasTkAgg(fig1, master=window1)
    toolbar1 = NavigationToolbar2Tk(canvas1, window1)
    toolbar1.update()
    canvas1.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)

    window2 = Toplevel(root)
    window2.geometry("1220x720")
    window2.title("Estimations")
    fig2, axes2 = plt.subplots(2, 2)
    canvas2 = FigureCanvasTkAgg(fig2, master=window2)
    toolbar2 = NavigationToolbar2Tk(canvas2, window2)
    toolbar2.update()
    canvas2.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)

    # Create the widget grid
    frame = Frame(root)
    frame.pack(side=TOP, fill=BOTH, expand=False)

    #some widgets get added in the first 6 rows of the frame's grid          

    #initialize grid
    grid = Frame(frame)
    grid.grid(sticky="news", column=0, row=7)
    frame.rowconfigure(7, weight=1)
    frame.columnconfigure(0, weight=1)

    rejectgrid = Frame(grid)


    # Créer le slider pour modifier le paramètre de la fonction
    
    slider_frame = Scale(frame, label="Frame", from_=1, to=frames, orient=HORIZONTAL, command=update_frame)
    slider_targets = Scale(frame, label="Number of targets", from_=1, to=MAXTARGETS, orient=HORIZONTAL, command=update_scatterplot)
    button_computeallframes = Button(frame, text="Compute all frames", command=compute_all_frames)
    button_clearoldpos = Button(frame, text="Clear poisitions trace", command=clear_pos)
    var_savepos = IntVar(frame)
    var_savepos.set(0)
    check_savepos = Checkbutton(frame, text='save pos data after computing', variable=var_savepos, onvalue=1, offvalue=0)
    
    
    options = ["antenna1","antenna2"]
    var_antenna = StringVar(frame)
    var_antenna.set(options[0])
    menu_antenna = OptionMenu(frame, var_antenna, *options, command=update_frame)

    options = ["WINDOW", "hann2d", "blackman2d", "none"]
    var_window = StringVar(frame)
    var_window.set(options[3])
    menu_window = OptionMenu(frame, var_window, *options, command=update_window)
    


    options = ["NOISEFILTER", "gauss2d", "none"]
    var_noisefilter = StringVar(frame)
    var_noisefilter.set(options[2])
    menu_noisefilter = OptionMenu(frame, var_noisefilter, *options, command=update_noisefilter)
    slider_noisefiltersize = Scale(frame, label="noise filter size", from_=1, to=N, orient=HORIZONTAL, command=update_noisefilter)
    slider_noisefiltersize.set(3)

    options = ["CORRFILTER", "sinc", "hsinc", "sinc1d", "gauss", "sinc(hann)", "none"]
    var_corrfilter = StringVar(frame)
    var_corrfilter.set(options[3])
    menu_corrfilter = OptionMenu(frame, var_corrfilter, *options, command=update_corrfilter)
    slider_corrfiltersize = Scale(frame, label="corr filter size", from_=1, to=N, orient=HORIZONTAL, command=update_corrfilter)
    slider_corrfiltersize.set(51)

    options = ["MAXMETHOD", "maxfilter", "h-max"]
    var_maxmethod = StringVar(frame)
    var_maxmethod.set(options[1])
    menu_maxmethod = OptionMenu(frame, var_maxmethod, *options, command=update_maxfilter)
    slider_maxmethodparameter = Scale(frame, label="max parameter", from_=0.0001, to=0.1, orient=HORIZONTAL, resolution=0.0001, command=update_maxfilter)
    slider_maxmethodsize = Scale(frame, label="max footprint size", from_=1, to=N, orient=HORIZONTAL, command=update_maxfilter)
    slider_maxmethodparameter.set(0.01)
    slider_maxmethodsize.set(7)

    options = ["ANGLE_EST","max", "arg"]
    var_angle_est = StringVar(frame)
    var_angle_est.set(options[2])
    menu_angle_est = OptionMenu(frame, var_angle_est, *options, command=update_scatterplot)

    options = ["PLOT1", "windowed", "prefiltered", "filtered", "maxfiltered"]
    var_plot1 = StringVar(frame)
    var_plot1.set(options[1])
    menu_plot1 = OptionMenu(frame, var_plot1, *options, command=update_plot1)
    var_marker00 = IntVar(frame)
    var_marker00.set(1)
    check_marker00 = Checkbutton(frame, text='plot1 tgt markers',variable=var_marker00, onvalue=1, offvalue=0, command=update_marker00)
    var_logplot1 = IntVar(frame)
    var_logplot1.set(1)
    check_logplot1 = Checkbutton(frame, text='plot1 log',variable=var_logplot1, onvalue=1, offvalue=0, command=update_plot1)

    options = ["PLOT2", "windowed", "prefiltered", "filtered", "maxfiltered"]
    var_plot2 = StringVar(frame)
    var_plot2.set(options[3])
    menu_plot2 = OptionMenu(frame, var_plot2, *options, command=update_plot2)
    var_marker01 = IntVar(frame)
    var_marker01.set(1)
    check_marker01 = Checkbutton(frame, text='plot2 tgt markers',variable=var_marker01, onvalue=1, offvalue=0, command=update_marker01)
    var_logplot2 = IntVar(frame)
    var_logplot2.set(1)
    check_logplot2 = Checkbutton(frame, text='plot2 log',variable=var_logplot2, onvalue=1, offvalue=0, command=update_plot2)

    options = ["SCATTERPLOT","yz"]
    var_scatter = StringVar(frame)
    var_scatter.set(options[1])
    menu_scatter = OptionMenu(frame, var_scatter, *options, command=update_scatterplot)

    options = ["FILTERPLOT", "noise", "corr", "maxfootprint"]
    var_filterplot = StringVar(frame)
    var_filterplot.set(options[2])
    menu_filterplot = OptionMenu(frame, var_filterplot, *options, command=update_filterplot)

    text_folder = Label(frame, text="Folder :"+FILENAME.split('\\')[-3]+"/"+FILENAME.split('\\')[-2])
    text_file = Label(frame, text="File :"+FILENAME.split('\\')[-1])
    slider_smoothing_param = Scale(frame, label="Tracking smoothing parameter", from_=1, to=30, orient=HORIZONTAL, resolution=1, command=update_tracks)
    options = ["raw", "smoothed"]
    var_estimplots = StringVar(frame)
    var_estimplots.set(options[0])
    menu_estimplots = OptionMenu(frame, var_estimplots, *options, command=update_tracks)
    
    options = ["raw smooth", "associate smooth"]
    var_tracking = StringVar(frame)
    var_tracking.set(options[0])
    menu_tracking = OptionMenu(frame, var_tracking, *options, command=update_tracks)

    label_RR = Label(rejectgrid, text="Range rejecting limits")
    entry_RRL = Entry(rejectgrid, bd =5)
    entry_RRH = Entry(rejectgrid, bd =5)
    entry_RRL.insert(0,-0.1)
    entry_RRH.insert(0,dmax+0.1)

    label_VR = Label(rejectgrid, text="Speed rejecting limits")
    entry_VRL = Entry(rejectgrid, bd =5)
    entry_VRH = Entry(rejectgrid, bd =5)
    entry_VRL.insert(0,-vmax-0.1)
    entry_VRH.insert(0,vmax+0.1)

    label_FR = Label(rejectgrid, text="Frame rejecting limits")
    entry_FRL = Entry(rejectgrid, bd =5)
    entry_FRH = Entry(rejectgrid, bd =5)
    entry_FRL.insert(0,1)
    entry_FRH.insert(0,frames)

    label_RR.grid(row=0, column=0,  sticky="news")
    entry_RRH.grid(row=0, column=2,  sticky="news")
    entry_RRL.grid(row=0, column=1,  sticky="news")

    label_VR.grid(row=1, column=0,  sticky="news")
    entry_VRH.grid(row=1, column=2,  sticky="news")
    entry_VRL.grid(row=1, column=1,  sticky="news")

    label_FR.grid(row=2, column=0,  sticky="news")
    entry_FRH.grid(row=2, column=2,  sticky="news")
    entry_FRL.grid(row=2, column=1,  sticky="news")

    # Afficher le slider et le menu déroulant dans la fenêtre Tkinter
    
    slider_frame.grid(              row=0, column=0, columnspan = 7, sticky="news")

    label_c0 = Label(frame, text="General")
    label_c0.grid(                  row=1, column=0,  sticky="news")
    menu_antenna.grid(              row=2, column=0,  sticky="news")
    slider_targets.grid(            row=3, column=0,  sticky="news")
    button_computeallframes.grid(   row=4, column=0,  sticky="news")
    check_savepos.grid(             row=5, column=0,  sticky="news")

    label_c1 = Label(frame, text="Maxima")
    label_c1.grid(                  row=1, column=1,  sticky="news")
    menu_maxmethod.grid(            row=2, column=1,  sticky="news")
    slider_maxmethodparameter.grid( row=3, column=1,  sticky="news")
    slider_maxmethodsize.grid(      row=4, column=1,  sticky="news")

    label_c2 = Label(frame, text="Noise")
    label_c2.grid(                  row=1, column=2,  sticky="news")
    menu_noisefilter.grid(          row=2, column=2,  sticky="news")
    slider_noisefiltersize.grid(    row=3, column=2,  sticky="news")

    label_c3 = Label(frame, text="Correlation")
    label_c3.grid(                  row=1, column=3,  sticky="news")
    menu_corrfilter.grid(           row=2, column=3,  sticky="news")
    slider_corrfiltersize.grid(     row=3, column=3,  sticky="news")

    label_c4 = Label(frame, text="Angle estimation")
    label_c4.grid(                  row=1, column=4,  sticky="news")
    menu_angle_est.grid(            row=2, column=4,  sticky="news")
    label_c42 = Label(frame, text="Windowing")
    label_c42.grid(                 row=3, column=4,  sticky="news")
    menu_window.grid(               row=4, column=4,  sticky="news")

    label_c5 = Label(frame, text="Plots")
    label_c5.grid(                  row=1, column=5, columnspan=2, sticky="news")
    menu_plot1.grid(                row=2, column=5,  sticky="news")
    menu_plot2.grid(                row=2, column=6,  sticky="news")
    menu_scatter.grid(              row=3, column=5,  sticky="news")
    menu_filterplot.grid(           row=3, column=6,  sticky="news")
    check_marker00.grid(            row=4, column=5,  sticky="news")
    check_marker01.grid(            row=4, column=6,  sticky="news")
    check_logplot1.grid(            row=5, column=5,  sticky="news")
    check_logplot2.grid(            row=5, column=6,  sticky="news")

    text_file.grid(                 row=0, column=8,  sticky="news")
    text_folder.grid(               row=1, column=8,  sticky="news")
    button_clearoldpos.grid(        row=2, column=8,  sticky="news")
    slider_smoothing_param.grid(    row=3, column=8,  sticky="news")
    menu_estimplots.grid(           row=4, column=8,  sticky="news")
    menu_tracking.grid(             row=5, column=8,  sticky="news")
    rejectgrid.grid(                row=6, column=8,  sticky="news")

    frame.columnconfigure(tuple(range(10)), weight=1)
    frame.rowconfigure(tuple(range(5)), weight=1)















    from matplotlib.colors import ListedColormap
    colors = plt.cm.jet(np.linspace(0, 1, 256))
    cmap = ListedColormap(colors)

    axes1[0,0].set_xlabel('speed[m/s]')
    axes1[0,0].set_ylabel('range[m]')
    axes1[0,0].set_ylim((0,dmax - R0))

    axes1[1,0].set_xlabel('speed[m/s]')
    axes1[1,0].set_ylabel('range[m]')
    axes1[1,0].set_ylim((0,dmax - R0))

    axes1[0,1].set_xlabel('speed[m/s]')
    axes1[0,1].set_ylabel('range[m]')
    axes1[0,1].set_ylim((0,dmax - R0))

    axes1[1,1].set_xlim(-dmax, dmax)
    axes1[1,1].set_ylim(0, dmax)
    axes1[1,1].set_xlabel('x')
    axes1[1,1].set_ylabel('y')
    axes1[1,1].set_title("Estimated position")


    axes2[0,0].set_title("Target angle")
    axes2[0,0].set_ylabel("angle [°]")
    axes2[0,0].set_xlabel("#frame")
    axes2[0,0].set_ylim((0,180))

    axes2[0,1].set_title("Target range")
    axes2[0,1].set_ylabel("range [m]")
    axes2[0,1].set_xlabel("#frame")
    axes2[0,1].set_ylim((-5,dmax))

    axes2[1,0].set_title("Target tracking")
    axes2[1,0].set_ylabel("y [m]")
    axes2[1,0].set_xlabel("x [m]")
    axes2[1,0].set_ylim((-5,dmax))
    axes2[1,0].set_xlim((-dmax,dmax))

    zeros = np.zeros((N,N))
    pcm1 = axes1[0,0].pcolormesh(v,d-R0,zeros,cmap=cmap)
    pcm2 = axes1[0,1].pcolormesh(v,d-R0,zeros,cmap=cmap)
    pcm3 = axes1[1,0].pcolormesh(v,d-R0,zeros,cmap=cmap)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.cm as cm
    divider = make_axes_locatable(axes1[0,0])
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    cb1 = fig1.colorbar(pcm1, cax=cax1, orientation='vertical')
    divider = make_axes_locatable(axes1[0,1])
    cax2 = divider.append_axes('right', size='5%', pad=0.05)
    cb2 = fig1.colorbar(pcm2, cax=cax2, orientation='vertical')

    divider = make_axes_locatable(axes1[1,0])
    cax3 = divider.append_axes('right', size='5%', pad=0.05)
    cb3 = fig1.colorbar(pcm3, cax=cax3, orientation='vertical')  

    targets_yz = []
    old_targets_yz = []
    markers00 = []
    markers01 = []

    angles = []
    ranges = []

    cols = ['r','g','b','purple','gray', 'orange']
    for i in range(MAXTARGETS):
        old_targets_yz.append(axes1[1,1].scatter(POSITIONS[i,:,0], POSITIONS[i,:,1], c=cols[i], alpha = 0.2, edgecolors='white', s=9))
        targets_yz.append(axes1[1,1].scatter(0,-10, c=cols[i], s=15))
        markers00.append(axes1[0,0].scatter(0,-10, edgecolors=cols[i], s=100, marker='X', facecolors='none'))
        markers01.append(axes1[0,1].scatter(0,-10, edgecolors=cols[i], s=100, marker='X', facecolors='none'))
        
        angles.append(axes2[0,0].scatter(ANGLES[i,:,0], ANGLES[i,:,1], c=cols[i], s=10))
        ranges.append(axes2[0,1].scatter(RANGES[i,:,0], RANGES[i,:,1],c=cols[i], s=10))



    
    update_window()
    root.mainloop()

if __name__ == "__main__":
    init(frames, N, Ms, Mc, v, d, R0, vmax, dmax, antenna1, antenna2, k, L, BETAS)