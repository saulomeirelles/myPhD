# rotarySpectrum - Gonela, Emery & Thompson methods to get rotary spectrum

def rotarySpectrum(x, y, NFFT, Fs, noverlap=0):

    from matplotlib.mlab import window_hanning
    import numpy as np
    from scipy.signal import detrend

    if len(x)<=NFFT:

        NFFT = int(2**np.ceil(np.log2(len(x))))
        n    = len(x)
        x    = np.resize(x, (NFFT,))

        x[n:] = 0
        y     = np.resize(y, (NFFT,))
        y[n:] = 0

    windowVals = window_hanning(np.ones((NFFT,),x.dtype))
    step       = NFFT - noverlap
    ind        = range(0,len(x)-NFFT+1,step)
    n          = len(ind)
    numFreqs   = int(NFFT//2 +1)

    Pxx = np.zeros((numFreqs,n), float)
    Pyy = np.zeros((numFreqs,n), float)
    Pxy = np.zeros((numFreqs,n), complex)
    Qxy = np.zeros((numFreqs,n), complex)
    Cxy = np.zeros((numFreqs,n), complex)

    for i in range(n):
        thisX    = x[ind[i]:ind[i]+NFFT]
        #thisX = windowVals*detrend(thisX)
        thisX    = detrend(thisX)
        fx       = np.absolute(np.fft.fft(thisX))**2 # density spectrum
        fxx      = np.fft.fft(thisX) # amplitude spectrum
        Pxx[:,i] = fx[:numFreqs]

        thisY    = y[ind[i]:ind[i]+NFFT]
        #thisY = windowVals*detrend(thisY)
        thisY    = detrend(thisY)
        fy       = np.absolute(np.fft.fft(thisY))**2
        fyy      = np.fft.fft(thisY)
        Pyy[:,i] = fy[:numFreqs]

        Pxy[:,i] = fyy[:numFreqs]*np.conjugate(fxx[:numFreqs]) # the cross-spectrum is the product between the amplitude spectra
        Qxy[:,i] = -np.imag(Pxy[:,i])
        Cxy[:,i] = np.real(Pxy[:,i])

    if n>1:
        Pxx = np.nanmean(Pxx,1)
        Pyy = np.nanmean(Pyy,1)
        Pxy = np.nanmean(Pxy,1)
        Qxy = np.nanmean(Qxy,1)
        Cxy = np.nanmean(Cxy,1)

    Pxx = np.divide(Pxx, np.linalg.norm(windowVals)**2) #,dtype='float')
    Pyy = np.divide(Pyy, np.linalg.norm(windowVals)**2) #,dtype='float')
    Pxy = np.divide(Pxy, np.linalg.norm(windowVals)**2) #,dtype='float')
    Qxy = np.divide(Qxy, np.linalg.norm(windowVals)**2) #,dtype='float')
    Cxy = np.divide(Cxy, np.linalg.norm(windowVals)**2) #,dtype='float')

    # rotary spectra (clockwise - counterclockwise)

    Rcw   = (Pxx + Pyy - 2 * Qxy) / 8.
    Rccw  = (Pxx + Pyy + 2 * Qxy) / 8.
    Rtot  = (Pxx + Pyy) / 4.

    # rotary coefficient

    Cr = (Rcw - Rccw) / Rtot  # - Gonella cw < 0  & ccw > 0 & unidirectional flow=0
    #Cr = (Rcw - Rccw) / (Rcw + Rccw) # - Emery & Thomson: cw > 0  & ccw < 0 & unidirectional flow=0

    # orientation of major axis

    R2 = (1 / 64.)*((Pxx + Pyy)**2 - 4*(Pxx*Pyy - Pxy**2))

    # mean orientation and stability

    E  = R2 / (Rcw*Rccw) # stability

    Phi = 0.5*np.arctan((np.pi / 180.)*(2*Pxy / (Pxx - Pyy)))

    # coherence

    Coh = (Pxy - 1j*Qxy) / np.sqrt(Pxx*Pyy)

    freqs = np.divide(Fs,NFFT,dtype=float)*np.arange(0,numFreqs)

    freqs = freqs.reshape( freqs.shape + (1,) ) # adding 2nd dimension

    return Pxx, Pyy, Pxy, Qxy, Cxy, Rcw, Rccw, Rtot, Cr, R2, E, Phi, Coh, freqs

# calcTurb - Shaw & Trawbridge method to split wave-induced velocities

def calcTurb(u1,v1,w1,u2,v2,w2,Tp,Fs):

    import numpy as np

    M     = len(u1)

    N     = np.round( ( Tp )*Fs )

    if not np.mod(N,2):
        N = N + 1

    A   = np.empty( ( int(M) , int(3*N) ) )

    U1  = np.array((u1,v1,w1)).T

    fac = (N - 1) / 2

    for row in np.arange(0,M):
        col         = np.linspace( row - fac, row + fac, N ).astype(int)
        col[col<1]  = 1
        col[col>=M] = M - 1
        U           = np.concatenate((u2[col], v2[col], w2[col]))
        A[row, : ]  = U

    h_hat   = np.dot( np.dot( np.linalg.inv( np.dot( A.T,A ) ), A.T ), U1 )

    U1_hat  = np.dot(A, h_hat)

    dU1_hat = U1 - U1_hat

    #uwCov   = np.cov( dU1_hat[:,0], w1 )
    #vwCov   = np.cov( dU1_hat[:,1], w1 )
    uwCov   = np.cov( dU1_hat[:,0], w1 )
    vwCov   = np.cov( dU1_hat[:,1], w1 )

    return U1_hat, dU1_hat, uwCov, vwCov, h_hat


def shearVelMadsen(ubr,omegar,ucr,zr,phiwc,zin):

    '''
    code for wave friction factor based on Madsen (1994)

    INPUTS:
    ubr:    wave orbital velocity
    omegar: angular wave frequency
    ucr:    current velocity at heigth zr
    zr:     reference heigth for current velocity
    phiwc:  angle between currents and waves at zr
    zin:    bottom roughness height

    OUTPUTS:
    ustarc: current friction velocity
    ustarw: wave maximum friction velocity
    ustarr: wave-current combined friction velocity
    fwc:    wave friction factor
    zoa:    apparent bottom roughness

    Author: Rafael

    Python version by: Saulo Meirelles

    '''

    import numpy as np

    MAXIT  = 20
    vkappa = 0.41
    fwc    = 0.4

    ustarc = np.empty((1,1))
    ustarw = np.empty((1,1))
    ustarr = np.empty((1,1))

    if np.abs(ubr) <= 0.01 and np.abs(ucr) <= 0.01:
        ustarc = 0.000001
        ustarw = 0.000001
        ustarr = 0.000001

    elif np.abs(ubr) <= 0.01:
        ustarc = ucr*vkappa/np.log(zr/zin)
        ustarw = 0.000001
        ustarr = ustarc

    cosphiwc = np.abs(np.cos(phiwc))
    kN       = 30*zin

    rmu     = np.empty((MAXIT,1))
    Cmu     = np.empty((MAXIT,1))
    fwci    = np.empty((MAXIT,1))
    ustarw2 = np.empty((MAXIT,1))
    ustarr2 = np.empty((MAXIT,1))
    ustarci = np.empty((MAXIT,1))
    dwc     = np.empty((MAXIT,1))

    rmu[0], Cmu[0]  = 0., 1.

    cukom = Cmu[0]*ubr/kN/omegar

    if 0.2 < cukom <= 100.:
        fwci[0] = Cmu[0]*np.exp(7.02*cukom**(-0.078)-8.82)
    elif 100. < cukom <= 10000.:
        fwci[0] = Cmu[0]*np.exp(5.61*cukom**(-0.109)-7.3)
    elif cukom > 10000.:
        fwci[0] = Cmu[0]*np.exp(5.61*10000**(-0.109)-7.3)
    else:
        fwci[0] = Cmu[0]*0.43

    ustarw2[0]  = 0.5*fwci[0]*ubr**2
    ustarr2[0]  = Cmu[0]*ustarw2[0]
    ustarr      = np.sqrt( ustarr2[0] )

    if cukom >= 8.0:
        dwc[0] = 2.0*vkappa*ustarr/omegar
    else:
        dwc[0] = kN


    lnzr       = np.log( zr / dwc[0] )
    lndw       = np.log( dwc[0] / zin )
    lnln       = lnzr / lndw
    bigsqr     = (-1.0) + np.sqrt( 1.0 + ( ( 4.0*vkappa*lndw ) / ( lnzr**2) )*(ucr / ustarr) )
    ustarci[0] = 0.5*ustarr*lnln*bigsqr

    i    = 0
    diff = 1

    while i < MAXIT & diff > 0.000005:

        i += 1

        rmu[i] = ustarci[i-1]**2 / ustarw2[i-1]
        Cmu[i] = np.sqrt( 1+2*rmu[i]*cosphiwc+rmu[i]**2 )

        cukom  = Cmu[i]*ubr / ( kN*omegar )

        if 0.2 < cukom <= 100:
            fwci[i] = Cmu[i]*np.exp(7.02*cukom**(-0.078)-8.82)
        elif 100 < cukom <= 10000:
            fwci[i] = Cmu[i]*np.exp(5.61*cukom**(-0.109)-7.3)
        elif cukom > 10000:
            fwci[i] = Cmu[i]*np.exp(5.61*10000**(-0.109)-7.3)
        else:
            fwci[i] = Cmu[i]*0.43


        ustarw2[i] = 0.5*fwci[i]*ubr**2
        ustarr2[i] = Cmu[i]*ustarw2[i]
        ustarr     = np.sqrt( ustarr2[i] )

        if cukom  >= 8.0:
            dwc[i] = 2.0*vkappa*ustarr/omegar
        else:
            dwc[i] = kN

        lnzr       = np.log( zr / dwc[i] )
        lndw       = np.log( dwc[i] / zin )
        lnln       = lnzr / lndw
        bigsqr     = (-1.0) + np.sqrt( 1.0+ ( ( 4.0*vkappa*lndw ) / ( lnzr**2) )*ucr / ustarr )
        ustarci[i] = 0.5*ustarr*lnln*bigsqr

        diff = np.abs( ( fwci[i]-fwci[i-1] ) / fwci[i] )


    ustarw = np.sqrt( ustarw2[i] )
    ustarc = ustarci[i]
    ustarr = np.sqrt( ustarr2[i] )
    zoa    = np.exp( np.log( dwc[i] )-( ustarc / ustarr )*np.log(dwc[i] / zin ) )

    if 'zoa' not in locals():
        zoa = np.nan
    elif np.isinf(zoa) == True:
        zoa = np.nan

    fwc = fwci[i]

    #print ustarc, ustarw, ustarr, fwc, zoa

    return ustarc, ustarw, ustarr, fwc, zoa


def read_ascii_adcp_FromVisea(fname):

    import datetime
    from pandas import DataFrame, concat
    from collections import OrderedDict
    import numpy as np

    #fname = 'MP14_ADCP_141017028t.000'

    f  = open(fname, 'r')
    fd = f.read().split('\n')

    main_header = dict()
    ens_header  = dict()
    ens         = dict()
    ens_data    = DataFrame()



    kw        = dict(columns=('depth',
        'vel_mag',
        'vel_dir',
        'east_vel','north_vel','up_vel',
        'error_vel',
        'abs1','abs2','abs3','abs4',
        'perc_good','discharge','longitude','latitude'))

    main_header['transect_ID'] = fname[-8:-4]

    try:
        # main header
        main_header['comment1'] = fd.pop(0)

        # main main_header
        main_header['comment2'] = fd.pop(0)

        # main main_header
        l = fd.pop(0).rstrip('\n').split()
        main_header['bin_size_cm'] = int(l[0])

        main_header['blank_cm'] = int(l[1])

        main_header['first_bin_cm'] = int(l[2])

        main_header['cells_number'] = int(l[3])

        main_header['ping_per_ens'] = int(l[4])

        main_header['time_per_ens'] = int(l[5])

        main_header['mode'] = int(l[6])

        cc = 0
        while True:

            # ensemble header line 01
            l = fd.pop(0).rstrip('\n').split()
            time = [int(i) for i in l[0:7]]
            time[0] += 2000  # in the source, they start counting in year 2000
            time[6] *= 10000 # 1/100 seconds to microseconds
            ens_header.setdefault('ens_time',[]).append(datetime.datetime(time[0], time[1], time[2], time[3], time[4], time[5], time[6]))

            ens_header.setdefault('ens_number',[]).append(int(l[7]))

            ens_header.setdefault('ens_per_segment',[]).append(l[8])

            ens_header.setdefault('pitch',[]).append(l[9])

            ens_header.setdefault('roll',[]).append(l[10])

            ens_header.setdefault('heading_corrected',[]).append(l[11])

            ens_header.setdefault('temp_adcp',[]).append(l[12])

            # ensemble header line 02
            l = fd.pop(0).rstrip('\n').split()
            ens_header.setdefault('east_vel_bt',[]).append(float(l[0])/100)

            ens_header.setdefault('north_vel_bt',[]).append(float(l[1])/100)

            ens_header.setdefault('up_vel_bt',[]).append(float(l[2])/100)

            ens_header.setdefault('error_vel_bt',[]).append(float(l[3])/100)

            ens_header.setdefault('east_vel_gga',[]).append(float(l[4])/100)

            ens_header.setdefault('north_vel_gga',[]).append(float(l[5])/100)

            ens_header.setdefault('up_vel_gga',[]).append(float(l[6])/100)

            ens_header.setdefault('error_vel_gga',[]).append(float(l[7])/100)

            depth = [float(i) for i in l[8:12]]

            ens_header.setdefault('depth_beam1',[]).append(depth[0])

            ens_header.setdefault('depth_beam2',[]).append(depth[1])

            ens_header.setdefault('depth_beam3',[]).append(depth[2])

            ens_header.setdefault('depth_beam4',[]).append(depth[3])

            ens_header.setdefault('depth_mean',[]).append(np.mean(depth))

            # ensemble header line 03
            l = fd.pop(0).rstrip('\n').split()
            ens_header.setdefault('elapsed_distance',[]).append(float(l[0]))

            ens_header.setdefault('elapsed_time',[]).append(float(l[1]))

            ens_header.setdefault('distance_north',[]).append(float(l[2]))

            ens_header.setdefault('distance_south',[]).append(float(l[3]))

            ens_header.setdefault('distance_good',[]).append(float(l[4]))

            # ensemble header line 04
            l = fd.pop(0).rstrip('\n').split()
            ens_header.setdefault('latitude',[]).append(float(l[0]))

            ens_header.setdefault('longitude',[]).append(float(l[1]))

            ens_header.setdefault('doNot_know1',[]).append(float(l[2]))

            ens_header.setdefault('doNot_know2',[]).append(float(l[3]))

            ens_header.setdefault('doNot_know3',[]).append(float(l[4]))

            # ensemble header line 05
            l = fd.pop(0).rstrip('\n').split()
            ens_header.setdefault('discharge_middle',[]).append(float(l[0]))

            ens_header.setdefault('discharge_top',[]).append(float(l[1]))

            ens_header.setdefault('discharge_bottom',[]).append(float(l[2]))

            ens_header.setdefault('discharge_middle',[]).append(float(l[0]))

            # ensemble header line 06
            l = fd.pop(0).rstrip('\n').split()

            ens_header.setdefault('number_beams_follow',[]).append(int(l[0]))

            ens_header.setdefault('unit_meas',[]).append(str(l[1]))

            ens_header.setdefault('vel_reference',[]).append(str(l[2]))

            ens_header.setdefault('unit_abs',[]).append((l[3]))

            data = []
            geopos = [str(ens_header['longitude'][cc]),str(ens_header['latitude'][cc])]

            for k in range(main_header['cells_number']):

                l = fd.pop(0).rstrip('\n').split()
                depth = l[0]

                l.extend(geopos)

                if len(l) == 15:

                    #if l[-3]  == '2147483647':

                        #l = transpose(['-99999' for i in l])

                    l[0] = depth
                    data.append(map(float,l))

                else:

                    l = np.transpose(['-32768' for i in range(15)])

                    l[0] = depth
                    data.append(map(float,l))


            #data = data[0:-1] # workaround - somehow data is duplicating the last row
            ens_dict = {ens_header['ens_number'][cc]: DataFrame(data,**kw)}
            ens.update(ens_dict)

            cc += 1

            if len(fd) <= 1:
                break

    finally:
        f.close()

    OrderedDict(sorted(ens.items(), key=lambda t: t[0]))

    #ens_data = concat(ens.values(),keys = ens.keys())

    return ens, ens_header, main_header

def ReadVertsFM_MapFile_oldvr(NetNode_x, NetNode_y, NetElemNode, m='none'):

    import numpy as np

    cor_x   = NetNode_x
    cor_y   = NetNode_y
    poly    = NetElemNode - 1 # 0-based Python indexes

    idx     = ~np.any(poly < 0., axis=1) # separetes trianges from quadangles

    if m=='none':

        verts1  = zip(cor_x[poly[idx,:]],
                                cor_y[poly[idx,:]]) # quadangles include all the 4 columns

        verts2  = zip(cor_x[poly[~idx,:3]],
                                cor_y[poly[~idx,:3]]) # triangles get the first 3 columns, the last one is just a flag (large negative number)
    else:

        x1, y1 = m(cor_x[poly[idx,:]], cor_y[poly[idx,:]])

        x2, y2 = m(cor_x[poly[~idx,:3]],cor_y[poly[~idx,:3]])

        verts1 = zip(x1,y1)
        verts2 = zip(x2,y2)


    verts1  = np.swapaxes(verts1, 1, 2)
    verts2  = np.swapaxes(verts2, 1, 2)

    verts   = list(verts1)+list(verts2)

    return verts, idx

def ReadVertsFM_MapFile_newvr(ds, m='none'):

    import numpy as np

    cor_x   = ds.NetNode_x.values
    cor_y   = ds.NetNode_y.values

    idx         = ~np.any( ds.NetElemNode.values.astype(int) < 0, axis=1 )


    idxqua     = (ds
                .NetElemNode
                .dropna('nNetElem')).values.astype(int) - 1 #~np.any(poly < 0., axis=1) # separetes trianges from quadangles
    idxtri     =(ds
                .NetElemNode
                .where(np.isnan(ds.NetElemNode.isel(nNetElemMaxNode=3))==True)
                .dropna('nNetElem',how='all')
                .dropna('nNetElemMaxNode',how='all')).values.astype(int) - 1

    if m=='none':

            verts1  = zip(cor_x[idxqua],
                          cor_y[idxqua]) # quadangles include all the 4 columns

            verts2  = zip(cor_x[idxtri],
                          cor_y[idxtri]) # triangles get the first 3 columns, the last one is just a flag (large negative number)


    else:

        x1, y1 = m(cor_x[idxqua], cor_y[idxqua])

        x2, y2 = m(cor_x[idxtri],cor_y[idxtri])

        verts1 = zip(x1,y1)
        verts2 = zip(x2,y2)


    verts1  = np.swapaxes(verts1, 1, 2)
    verts2  = np.swapaxes(verts2, 1, 2)

    verts   = list(verts1)+list(verts2)

    return verts, idx

def rotation2D_velocities(u,v,angleDeg=42):

    import numpy as np

    RotAngle  = np.deg2rad(angleDeg)
    RotMatrix = np.array( [[ np.cos(RotAngle),  -np.sin(RotAngle)] ,
                           [ np.sin(RotAngle),  np.cos(RotAngle)]] )

    u_vec     = np.reshape(u, (1, np.size(u) ) )
    v_vec     = np.reshape(v, (1, np.size(v) ) )

    VelMatrix = np.array( [u_vec[0][:] , v_vec[0][:]] )
    VelRot    = np.dot(RotMatrix,VelMatrix)

    u_rot     = VelRot[0,:]
    v_rot     = VelRot[1,:]

    u_rot     = np.reshape(u_rot, ( np.shape(u) ) )
    v_rot     = np.reshape(v_rot, ( np.shape(v) ) )

    return u_rot, v_rot

def reject_outliers(data, m = 2.):
    import numpy as np

    d    = np.abs(data - np.median(data))
    mdev = np.median(d)
    s    = d/mdev if mdev else 0.
    return s<m #data[s<m]

def swirl_strength(u,w,x,z, v=None):

    '''
    By Max Rademacher (Matlab)
    '''

    import numpy as np
    from scipy import signal

    convmatx = np.array([[0., 0., 0.],[1., 0., -1.],[0., 0., 0.]])
    convmatz = np.array([[0., 1., 0.],[0., 0., 0.], [0., -1., 0.]])

    dudz  = signal.convolve2d(u,convmatz,mode='valid') / signal.convolve2d(z,convmatz,mode='valid')
    if v:
        dvdz  = signal.convolve2d(v,convmatz,mode='valid') / signal.convolve2d(z,convmatz,mode='valid')
    dudx  = signal.convolve2d(u,convmatx,mode='valid') / signal.convolve2d(x,convmatx,mode='valid')
    dwdz  = signal.convolve2d(w,convmatz,mode='valid') / signal.convolve2d(z,convmatz,mode='valid')
    dwdx  = signal.convolve2d(w,convmatx,mode='valid') / signal.convolve2d(x,convmatx,mode='valid')

    #curly =  (dudz - dwdx)
    curly =  (dudz)  # ignoring vertical vels

    SW    = np.empty(np.shape(dudx))

    for i in range(np.shape(dudx)[0]):
        for j in range(np.shape(dudx)[1]):
            A                = np.array( [ [dudx[i,j], dudz[i,j] ], [dwdx[i,j], dwdz[i,j] ] ] )
            A[np.isnan(A)]   = 0.
            A[np.isinf(A)]   = 0.
            SR = (A + A.T)/2 #Strain rate tensor
            OR = (A - A.T)/2 #Vorticity tensor
            eigenvalues,_    = np.linalg.eig(A)
            SW[i,j]          = np.max(np.unique(np.abs(np.imag(eigenvalues))))

    if v:
        return SW, dudz, dudx, curly, SR, OR, dvdz
    else:
        return SW, dudz, dudx, curly, SR, OR

def get_radar_times(datadir):

    '''
    By Max Rademacher
    '''
    import gdal
    import pyproj
    import glob
    import os
    import numpy as np
    import re
    import datetime
    from scipy import ndimage

    imlist = glob.glob(os.path.join(datadir,'*.tif'))
    datvec = []

    for im in imlist:
        path, filename = os.path.split(im)
        tstr = re.findall('[0-9]{4}.*(?=UTC)', filename)[0]
        datvec.append(datetime.datetime.strptime(tstr,'%Y-%m-%d %H.%M.%S'))

    return datvec

def get_radar_tiff(datadir,t):

    '''
    By Max Rademacher
    '''
    import gdal
    import pyproj
    import glob
    import os
    import numpy as np
    import re
    import datetime
    from scipy import ndimage

    ds = gdal.Open(os.path.join(datadir,datetime.datetime.strftime(t,'%Y-%m-%d %H.%M.%S') + 'UTC.tif'))
    band = ds.GetRasterBand(1)
    img = band.ReadAsArray()
    nrows, ncols = img.shape
    x0, dx, dxdy, y0, dydx, dy = ds.GetGeoTransform()
    x = np.arange(x0+dx/2,x0-dx/2+ncols*dx,dx)
    y = np.arange(y0,y0+nrows*dy,dy)
    X,Y = np.meshgrid(x,y)

    v,u = img.shape
    v = np.arange(v)
    u = np.arange(u)
    [u,v] = np.meshgrid(u,v)

    uc = 511.5
    vc = 511.5
    r = 503
    dst = np.sqrt((u-uc)**2+(v-vc)**2)
    ind = dst > r
    img[ind] = 255
    img = img[9:1015,9:1015]
    X = X[9:1015,9:1015]
    Y = Y[9:1015,9:1015]

    coorin=pyproj.Proj("+init=EPSG:32631")
    coorout=pyproj.Proj("+init=EPSG:28992")
    Xc,Yc = pyproj.transform(coorin,coorout,X,Y)

    WGS84 = pyproj.Proj("+init=EPSG:4326")
    RD = pyproj.Proj("+init=EPSG:28992")
    loni, lati = pyproj.transform(RD, WGS84, Xc, Yc)

    return loni, lati, Xc,Yc,img

def deg_to_dist(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    http://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
    """
    from math import radians, cos, sin, asin, sqrt

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2.)**2 + cos(lat1) * cos(lat2) * sin(dlon/2.)**2
    c = 2. * asin(sqrt(a))
    km = 6367. * c
    return km

def nearestDate(dates, pivot):
    return min(dates, key=lambda x: abs(x - pivot))


#http://stackoverflow.com/questions/8776414/python-datetime-to-matlab-datenum
def datetime2matlabdn(date):

    import datetime as dt

    mdn               = date  + dt.timedelta(days = 366)
    frac_seconds      = (date - dt.datetime(date.year,date.month,date.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
    frac_microseconds = date.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    return mdn.toordinal() + frac_seconds + frac_microseconds

def matlabDatenum2Datetime(datenum):

    import datetime as dt

    pydate  = [dt.datetime.fromordinal(int(i)) + dt.timedelta(days = i %1) - dt.timedelta(days = 366) for i in datenum]
    yearday = [(yd - dt.datetime(yd.year, 1, 1)).total_seconds()/(60.*60.*24) + 1 for yd in pydate] # datetime does that already

    return pydate, yearday


def lon_lat_to_cartesian(lon, lat, R = 1):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R
    http://earthpy.org/interpolation_between_grids_with_ckdtree.html
    """
    import numpy as np

    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x =  R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x,y,z

# http://web.mit.edu/bgolder/www/11.521/idw/
# http://stackoverflow.com/questions/3104781/inverse-distance-weighted-idw-interpolation-with-python
def inverse_distance_weighting(x,y,z,x_grid,y_grid,neighbor_limit=10):

    from scipy.spatial import cKDTree
    import numpy as np

    tree       = cKDTree(zip(x,y),leafsize=50)
    d, inds    = tree.query(zip(x_grid,y_grid), k= neighbor_limit, p=2)
    w          = 1.0 / d**2
    z_idw      = np.sum(w * z[inds], axis=1) / np.sum(w, axis=1)

    return z_idw

def cart2pol(x, y):
    import numpy as np
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    import numpy as np
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

############################################################




############################################################

def plotSandMotor(ax, fill=0, angle=0, fillcolor='white', zorder=10):
    import scipy.io as spio
    import numpy as np
    import cartopy.crs as ccrs

    def loadmat(filename):
        '''
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        '''
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return _check_keys(data)

    def _check_keys(dict):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in dict:
            if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
                dict[key] = _todict(dict[key])
        return dict

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                dict[strg] = _todict(elem)
            else:
                dict[strg] = elem
        return dict

    import pyproj
    import platform

    if platform.system()=='Windows':
        z = loadmat(r"d:\00_PhD\Dataset\MP14\01_ADCP_13hour\ADCP20141017_proc26Nov\Zandmotor_XYZ_RDNAP_2014_09.mat")
    if platform.system()=='Linux':
        z = loadmat(r"/media/saulo/SAULO5TB/00_PhD/Dataset/MP14/01_ADCP_13hour/ADCP20141017_proc26Nov/Zandmotor_XYZ_RDNAP_2014_09.mat")


    depth = z['XYZ']['Z']
    x     = z['XYZ']['X']
    y     = z['XYZ']['Y']

    WGS84 = pyproj.Proj(init='EPSG:4326')
    RD = pyproj.Proj(init='EPSG:28992')
    zlon, zlat = pyproj.transform(RD, WGS84, x, y)

    from matplotlib.mlab import griddata

    def grid(x, y, z, resX=50, resY=50):
        "Convert 3 column data to matplotlib grid"
        xi = np.linspace(min(x), max(x), resX)
        yi = np.linspace(min(y), max(y), resY)
        Z = griddata(x, y, z, xi, yi, interp='linear')
        X, Y = np.meshgrid(xi, yi)
        return X, Y, Z

    if angle!=0:

        RotAngle  = np.deg2rad(angle)
        RotMatrix = np.array( [[ np.cos(RotAngle),  -np.sin(RotAngle)] ,
                               [ np.sin(RotAngle),  np.cos(RotAngle)]] )

        original_Matrix = np.array( [zlon, zlat] )

        rotated_Matrix  = np.dot( RotMatrix, original_Matrix )

        zlon  = rotated_Matrix[0,:]
        zlat  = rotated_Matrix[1,:]

    xx, yy, zz = grid(zlon[::30],zlat[::30],depth[::30])

    if fill==0:

        ax.contour(xx,yy,zz,levels=[1, 2],colors='grey', linewidths=3)

        CS = ax.contour(xx,yy,zz,levels=[-10, -8, 0],colors='black')
        ax.clabel(CS, fontsize=22, inline=1, fmt='%d')

    elif fill==1:

        ax.contourf(xx,yy,zz, levels=np.arange(0,10,1), colors=fillcolor, zorder=zorder )
        ax.contour(xx,yy,zz,levels=[0, 1, 2],colors='black', linewidths=3, zorder=zorder+1 )
        CS = ax.contour(xx,yy,zz,levels=[-10, -8],colors='black')
        ax.clabel(CS, fontsize=18, inline=1, fmt='%d')



    return  zlon, zlat, depth

def compass(u, v, ax, arrowprops=None):


    import matplotlib.pyplot as plt
    import cart2pol


    """
    Compass draws a graph that displays the vectors with
    components `u` and `v` as arrows from the origin.

    Examples
    --------
    >>> import numpy as np
    >>> u = [+0, +0.5, -0.50, -0.90]
    >>> v = [+1, +0.5, -0.45, +0.85]
    >>> compass(u, v)
    https://ocefpaf.github.io/python4oceanographers/blog/2015/02/09/compass/
    """

    angles, radii = cart2pol(u, v)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    kw = dict(arrowstyle="->", color='k')
    if arrowprops:
        kw.update(arrowprops)
    [ax.annotate("", xy=(angle, radius), xytext=(0, 0),
                 arrowprops=kw) for
     angle, radius in zip(angles, radii)]

    ax.set_ylim(0, np.max(radii))

    return fig, ax


def plotOgives(wavenumb, ogives_UW_VW, outfile):

    import matplotlib.pyplot as plt

    fs       = 16
    lw       = 2
    filename = str(outfile)

    fig, ax  = plt.subplots(1, 2, figsize=(5,3), sharey=True )

    for k in range(2):

        ax[k].semilogx(wavenumb[1:], ogives_UW_VW[:,k], '-k', lw=lw)

        ax[k].set_xlim(1e-1,1e1)
        ax[k].set_ylim(-0.6,1.6)

        ax[k].set_xlabel('$2\pi fz / V$', fontsize=fs)
        if k == 0:
            ax[k].set_ylabel( '$Og_{u^{\prime} w^{\prime}} (f)$', fontsize=fs )
        else:
            ax[k].set_ylabel( '$Og_{v^{\prime} w^{\prime}} (f)$', fontsize=fs )

        ax[k].grid(True,which="both")
        ax[k].tick_params(axis='both', which='major', labelsize=fs)

        fig.tight_layout()

        fig.savefig( filename + '.png' )

    plt.close()

    return


def plotFilterWeight(Tp, h_hat, outfile):

    import matplotlib.pyplot as plt
    import numpy as np
    from pandas import rolling_mean

    filename = str(outfile)

    lag     = np.linspace( -Tp/2/2, Tp/2/2, len(h_hat)/3 )

    wdw     = len(lag)/5
    fs      = 16

    huu     = h_hat[0:len(h_hat)/3, 0]
    huv     = h_hat[0:len(h_hat)/3, 1]
    huw     = h_hat[0:len(h_hat)/3, 2]

    huu_avg = rolling_mean( huu, wdw, min_periods=0 )
    huv_avg = rolling_mean( huv, wdw, min_periods=0 )
    huw_avg = rolling_mean( huw, wdw, min_periods=0 )

    hVars   = np.array([ huu, huv, huw ])
    haVars  = np.array([ huu_avg, huv_avg, huw_avg ])
    ylabel  = np.array(['$\hat{h}_{uu}$','$\hat{h}_{uv}$','$\hat{h}_{uw}$'])


    fig, ax = plt.subplots(1,3, figsize = (20,6), sharey=True)

    for k in range(3):

        ax[k].plot(lag,hVars[k], '-k')
        ax[k].plot(lag,haVars[k],'-r')
        ax[k].plot(lag,haVars[k]*0,'--k')

        ax[k].set_ylabel(ylabel[k], fontsize = 20)
        ax[k].tick_params(axis='both', which='major', labelsize=fs)
        ax[k].grid()

        fig.tight_layout()

        fig.savefig( filename + '.png' )

    plt.close()

    return

    """
Python translation of Zhigang Xu's tidal_ellipse MATLAB tools, available at
http://woodshole.er.usgs.gov/operations/sea-mat/tidal_ellipse-html/ap2ep.html

Converted to Python by Pierre Cazenave (Plymouth Marine Laboratory), October
2012. Email: pica@pml.ac.uk

Authorship Copyright:

   The author retains the copyright of this program, while you are welcome
to use and distribute it as long as you credit the author properly and
respect the program name itself. Particularly, you are expected to retain the
original author's name in this original version or any of its modified version
that you might make. You are also expected not to essentially change the name
of the programs except for adding possible extension for your own version you
might create, e.g. ap2ep_xx is acceptable.  Any suggestions are welcome and
enjoy my program(s)!


Author Info:
_______________________________________________________________________________

  Zhigang Xu, Ph.D.
  (pronounced as Tsi Gahng Hsu)
  Research Scientist
  Coastal Circulation
  Bedford Institute of Oceanography
  1 Challenge Dr.
  P.O. Box 1006                    Phone  (902) 426-2307 (o)
  Dartmouth, Nova Scotia           Fax    (902) 426-7827
  CANADA B2Y 4A2                   email xuz@dfo-mpo.gc.ca
_______________________________________________________________________________

Release Date: Nov. 2000, Revised on May. 2002 to adopt Foreman's northern semi
major axis convention.

"""

import numpy as np
import matplotlib.pyplot as plt


def ap2ep(Au, PHIu, Av, PHIv, plot_demo=False):
    """
    Convert tidal amplitude and phase lag (ap-) parameters into tidal ellipse
    (ep-) parameters. Please refer to ep2ap for its inverse function.

    Usage:

    SEMA, ECC, INC, PHA, w = ap2ep(Au, PHIu, Av, PHIv, plot_demo=False)

    Where:

        Au, PHIu, Av, PHIv are the amplitudes and phase lags (in degrees) of
        u- and v- tidal current components. They can be vectors or
        matrices or multidimensional arrays.

        plot_demo is an optional argument, when it is supplied as an array
        of indices, say [i j k l], the program will plot an ellipse
        corresponding to Au[i, j, k, l], PHIu[i, j, k, l], Av[i, j, k, l], and
        PHIv[i, j, k, l]. Defaults to False (i.e. no plot).

        Any number of dimensions are allowed as long as your computer
        resource can handle.

        SEMA: Semi-major axes, or the maximum speed.

        ECC:  Eccentricity, the ratio of semi-minor axis over the semi-major
              axis; its negative value indicates that the ellipse is traversed
              in clockwise direction.

        INC:  Inclination, the angles (in degrees) between the semi-major axes
              and u-axis.

        PHA:  Phase angles, the time (in angles and in degrees) when the tidal
              currents reach their maximum speeds,  (i.e.  PHA=omega*tmax).

              These four ep-parameters will have the same dimensionality (i.e.,
              vectors, or matrices) as the input ap-parameters.

        w:    A matrix whose rows allow for plotting ellipses and whose columns
              are for different ellipses corresponding columnwise to SEMA. For
              example, plot(np.real(w[0, :]), np.imag(w[0, :])) will let you
              see the first ellipse. You may need to use squeeze function when
              w is a more than two dimensional array. See example.py.

    Document:   tidal_ellipse.ps

    Revisions: May  2002, by Zhigang Xu,  --- adopting Foreman's northern semi
    major axis convention.

    For a given ellipse, its semi-major axis is undetermined by 180. If we
    borrow Foreman's terminology to call a semi major axis whose direction lies
    in a range of [0, 180) as the northern semi-major axis and otherwise as a
    southern semi major axis, one has freedom to pick up either northern or
    southern one as the semi major axis without affecting anything else.
    Foreman (1977) resolves the ambiguity by always taking the northern one as
    the semi-major axis. This revision is made to adopt Foreman's convention.
    Note the definition of the phase, PHA, is still defined as the angle
    between the initial current vector, but when converted into the maximum
    current time, it may not give the time when the maximum current first
    happens; it may give the second time that the current reaches the maximum
    (obviously, the 1st and 2nd maximum current times are half tidal period
    apart) depending on where the initial current vector happen to be and its
    rotating sense.

    Version 2, May 2002

    Converted to Python by Pierre Cazenave, October 2012.

    Authorship Copyright:

       The author retains the copyright of this program, while  you are welcome
    to use and distribute it as long as you credit the author properly and
    respect the program name itself. Particularly, you are expected to retain
    the original author's name in this original version or any of its modified
    version that you might make. You are also expected not to essentially
    change the name of the programs except for adding possible extension for
    your own version you might create, e.g. ap2ep_xx is acceptable.  Any
    suggestions are welcome and enjoy my program(s)!


    Author Info:
    _______________________________________________________________________
      Zhigang Xu, Ph.D.
      (pronounced as Tsi Gahng Hsu)
      Research Scientist
      Coastal Circulation
      Bedford Institute of Oceanography
      1 Challenge Dr.
      P.O. Box 1006                    Phone  (902) 426-2307 (o)
      Dartmouth, Nova Scotia           Fax    (902) 426-7827
      CANADA B2Y 4A2                   email xuz@dfo-mpo.gc.ca
    _______________________________________________________________________

    Release Date: Nov. 2000, Revised on May. 2002 to adopt Foreman's northern
    semi major axis convention.

    """

    # Assume the input phase lags are in degrees and convert them in radians.
    PHIu = PHIu / 180 * np.pi
    PHIv = PHIv / 180 * np.pi

    # Make complex amplitudes for u and v
    i = 1j
    u = Au * np.exp(-i * PHIu)
    v = Av * np.exp(-i * PHIv)

    # Calculate complex radius of anticlockwise and clockwise circles:
    wp = (u + i * v) / 2           # for anticlockwise circles
    wm = np.conj(u - i * v) / 2    # for clockwise circles
    # and their amplitudes and angles
    Wp = np.abs(wp)
    Wm = np.abs(wm)
    THETAp = np.angle(wp)
    THETAm = np.angle(wm)

    # calculate ep-parameters (ellipse parameters)
    SEMA = Wp + Wm                 # Semi Major Axis, or maximum speed
    SEMI = Wp - Wm                 # Semi Minor Axis, or minimum speed
    ECC = SEMI / SEMA              # Eccentricity

    PHA = (THETAm - THETAp) / 2    # Phase angle, the time (in angle) when
                                   # the velocity reaches the maximum
    INC = (THETAm + THETAp) / 2    # Inclination, the angle between the
                                   # semi major axis and x-axis (or u-axis).

    # convert to degrees for output
    PHA = PHA / np.pi*180
    INC = INC / np.pi*180
    THETAp = THETAp / np.pi*180
    THETAm = THETAm / np.pi*180

    # map the resultant angles to the range of [0, 360].
    PHA = np.mod(PHA + 360, 360)
    INC = np.mod(INC + 360, 360)

    # Mar. 2, 2002 Revision by Zhigang Xu    (REVISION_1)
    # Change the southern major axes to northern major axes to conform the tidal
    # analysis convention  (cf. Foreman, 1977, p. 13, Manual For Tidal Currents
    # Analysis Prediction, available in www.ios.bc.ca/ios/osap/people/foreman.htm)
    k = np.fix(INC / 180)
    INC = INC - k * 180
    PHA = PHA + k * 180
    PHA = np.mod(PHA, 360)

    if plot_demo:
        plot_ell(SEMA, ECC, INC, PHA, plot_demo)

    ndot = np.prod(np.shape(SEMA))
    dot = 2 * np.pi / ndot
    ot = np.arange(0, 2 * np.pi, dot)
    w = wp.flatten() * np.exp(i * ot) + wm.flatten() * np.exp(-i * ot)
    w = np.reshape(w, np.shape(wp))

    return SEMA, ECC, INC, PHA, w


def ep2ap(SEMA, ECC, INC, PHA, plot_demo=False):
    """
    Convert tidal ellipse parameters into amplitude and phase lag parameters.
    Its inverse is app2ep.m. Please refer to app2ep for the meaning of the
    inputs and outputs.

    Zhigang Xu
    Oct. 20, 2000

    Converted to Python by Pierre Cazenave, October 2012.

    Authorship Copyright:

        The author of this program retains the copyright of this program, while
    you are welcome to use and distribute this program as long as you credit
    the author properly and respect the program name itself. Particularly, you
    are expected to retain the original author's name in this original version
    of the program or any of its modified version that you might make.  You are
    also expected not to essentially change the name of the programs except for
    adding possible extension for your own version you might create, e.g.
    app2ep_xx is acceptable.  Any suggestions are welcome and enjoy my
    program(s)!


    Author Info:
    _______________________________________________________________________
      Zhigang Xu, Ph.D.
      (pronounced as Tsi Gahng Hsu)
      Research Scientist
      Coastal Circulation
      Bedford Institute of Oceanography
      1 Challenge Dr.
      P.O. Box 1006                    Phone  (902) 426-2307 (o)
      Dartmouth, Nova Scotia           Fax    (902) 426-7827
      CANADA B2Y 4A2                   email xuz@dfo-mpo.gc.ca
    _______________________________________________________________________

    Release Date: Nov. 2000

    """

    i = 1j

    Wp = (1 + ECC) / 2 * SEMA
    Wm = (1 - ECC) / 2 * SEMA
    THETAp = INC - PHA
    THETAm = INC + PHA

    # Convert degrees into radians
    THETAp = THETAp / 180 * np.pi
    THETAm = THETAm / 180 * np.pi

    # Calculate wp and wm.
    wp = Wp * np.exp(i * THETAp)
    wm = Wm * np.exp(i * THETAm)

    ndot = np.prod(np.shape(SEMA))
    dot = 2 * np.pi / ndot
    ot = np.arange(0, 2 * np.pi, dot)
    w = wp.flatten() * np.exp(i * ot) + wm.flatten() * np.exp(-i * ot)
    w = np.reshape(w, np.shape(wp))

    # Calculate cAu, cAv --- complex amplitude of u and v
    cAu = wp + np.conj(wm)
    cAv = -i * (wp-np.conj(wm))
    Au = np.abs(cAu)
    Av = np.abs(cAv)
    PHIu = -np.angle(cAu) * 180 / np.pi
    PHIv = -np.angle(cAv) * 180 / np.pi

    # flip angles in the range of [-180 0) to the range of [180 360).
    id = PHIu < 0
    PHIu[id] = PHIu[id] + 360
    id = PHIv < 0
    PHIv[id] = PHIv[id] + 360

    if plot_demo:
        plot_ell(SEMA, ECC, INC, PHA, plot_demo)

    return Au, PHIu, Av, PHIv, w


def cBEpm(g, f, sigma, nu, kappa, z, h):
    """
    Evaluate the theoretical vertical profiles (or Bottom Ekman spiral
    profiles) of tidal currents in the two rotary directions driven by
    half-unit of sea surface gradients in the two directions respectively. Eddy
    viscosity is assumed as vertically invariant. See tidal_ellipse.ps for more
    details.

    Inputs:

        g:      acceleration gravity
        f:      the Coriolis parameter
        nu:     the eddy viscosity
        kappa:  the bottom frictional coefficient
        z:      the vertical coordinates, can be a vector but must be
                within [0 -h];
        h:      the water depth, must be positive.

        Note: except for z, all other inputs must be scalars.

    Outputs:

        BEp and BEm, the same dimensions of z,  the outputs for the vertical
            velocity profiles driven respectively by a unit of sea surface
            slope in the positive rotation direction and negative rotation
            direction for when the eddy viscosity is vertically invariant. See
            the associated document for more details.

    Authorship Copyright:

       The author of this program retains the copyright of this program, while
    you are welcome to use and distribute this program as long as you credit
    the author properly and respect the program name itself. Particularly,
    you are expected to retain the original author's name in this original
    version of the program or any of its modified version that you might make.
    You are also expected not to essentially change the name of the programs
    except for adding possible extension for your own version you might create,
    e.g. ap2ep_xx is acceptable.  Any suggestions are welcome and enjoying my
    program(s)!


    Author Info:
    _______________________________________________________________________
      Zhigang Xu, Ph.D.
      (pronounced as Tsi Gahng Hsu)
      Research Scientist
      Coastal Circulation
      Bedford Institute of Oceanography
      1 Challenge Dr.
      P.O. Box 1006                    Phone  (902) 426-2307 (o)
      Dartmouth, Nova Scotia           Fax    (902) 426-7827
      CANADA B2Y 4A2                   email zhigangx@emerald.bio.dfo.ca
                                             zhigang_xu_98@yahoo.com
    _______________________________________________________________________

    Release Date: Nov. 2000

    """

    if (len(g) > 1) | (len(f) > 1) | (len(sigma) > 1) | \
            (len(nu) > 1) | (len(kappa) > 1) | (len(h) > 1):
        print('inputs of g, f, sigma, nu, kappa, and h should be all scalars!')
        raise

    if (any(z / h > 0)) | (any(z / h < -1)):
        print('z must be negative and must be within [0 -h]')

    delta_e = np.sqrt(2 * nu / f)  # Ekman depth
    alpha = (1 + 1j) / delta_e * np.sqrt(1 + sigma / f)
    beta = (1 + 1j) / delta_e * np.sqrt(1 - sigma / f)

    BEp = get_BE(g, alpha, h, z, nu, kappa)
    BEm = get_BE(g, beta, h, z, nu, kappa)

    return BEp, BEm


def get_BE(g, alpha, h, z, nu, kappa):
    """ Child function of cBEpm """

    z = z.flatten()
    z_h = z / h
    ah = alpha * h
    az = alpha * z
    ah2 = ah * 2
    anu_k = alpha * nu / kappa
    nu_kh = nu / (kappa * h)

    # Series solution
    if abs(ah) < 1:
        T = 10
        C = -g * h * h / (nu * (1 + anu_k * np.tanh(ah))) * 2
        A1 = (1 - z_h * z_h) / 2 + nu_kh
        B1 = np.exp(-ah) / (1 + np.exp(-ah2))
        B = B1
        series_sum = A1 * B1

        for t in np.arange(2, T):
            t2 = 2*t
            A = (1 - z_h**t2) / t2 + nu_kh
            B = B * ah * ah / (t2 - 1) / (t2 - 2)
            series_sum = series_sum + A * B

        BE = C*series_sum

    # Finite solution
    else:
        c = -g * h * h / nu
        denom = (np.exp(az - ah) + np.exp(-(az + ah))) / (1 + np.exp(-2 * ah))

        numer = 1 + anu_k * np.tanh(ah)
        BE = c * ((1 - denom / numer) / (ah * ah))

    return BE


def sub2ind(shape, pos):
    """
    Substitute of MATLAB's sub2ind function for NumPy.

    t = numpy.random.random([2, 4, 5, 2])
    n = sub2ind(numpy.shape(t), [1, 2, 4, 1])
    >>> n
    69

    From http://stackoverflow.com/questions/4114461

    """
    res = 0
    acc = 1
    for pi, si in zip(reversed(pos), reversed(shape)):
        res += pi * acc
        acc *= si

    return res


def plot_ell(SEMA, ECC, INC, PHA, IND=[1]):
    """
    An auxiliary function used in ap2ep and ep2ap for plotting tidal ellipse.
    The inputs, MA, ECC, INC and PHA are the output of ap2ep and IND is a
    vector for indices for plotting a particular ellipse, e.g., if IND=[2 3 1]
    the ellipse corresponding to the indices of [2,3,1] will be plotted.

    By default, the first ellipse is always plotted.

    Converted to Python by Pierre Cazenave, October 2012.

    """

    len_IND = len(IND)
    if IND:
        cmd = 'sub2ind(size_SEMA, '
        if len_IND == 1:
            titletxt = 'Ellipse '
        else:
            titletxt = 'Ellipse ('

        for k in range(len_IND):
            if k == 0:
                cmd = cmd + '[' + str(IND[k])
            else:
                cmd = cmd + ',' + str(IND[k])

            if k < len_IND-1:
                titletxt = titletxt + str(IND[k]) + ','
            elif len_IND == 1:
                titletxt = titletxt + str(IND[k])
            else:
                titletxt = titletxt + str(IND[k]) + ')'

        cmd = 'n = ' + cmd + '])'
        # This is pretty nasty, but it works.
        exec(cmd)

        plt.gcf()
        plt.clf()
        do_the_plot(SEMA.flatten()[n], ECC.flatten()[n], INC.flatten()[n], PHA.flatten()[n])
        titletxt = titletxt + ',  (red) green (anti-) clockwise component'
        plt.title(titletxt)
    elif len_IND:
        print('IND input contains zero element(s)!\nNo ellipse will be plotted.')


def do_the_plot(SEMA, ECC, INC, PHA):
    """
    Ellipse plot subfunction.

    Converted to Python by Pierre Cazenave, October 2012.

    """

    i = 1j

    SEMI = SEMA * ECC
    Wp = (1 + ECC) / 2 * SEMA
    Wm = (1 - ECC) / 2 * SEMA
    THETAp = INC - PHA
    THETAm = INC + PHA

    # Convert degrees into radians
    THETAp = THETAp / 180 * np.pi
    THETAm = THETAm / 180 * np.pi
    INC = INC / 180 * np.pi
    PHA = PHA / 180 * np.pi

    # Calculate wp and wm.
    wp = Wp * np.exp(i * THETAp)
    wm = Wm * np.exp(i * THETAm)

    dot = np.pi / 36
    ot = np.arange(0, 2 * np.pi, dot)
    a = wp * np.exp(i * ot)
    b = wm * np.exp(-i * ot)
    w = a + b

    wmax = SEMA * np.exp(i * INC)
    wmin = SEMI * np.exp(i * (INC + np.pi / 2))

    plt.plot(np.real(w), np.imag(w))
    plt.axis('equal')
    plt.hold('on')
    plt.plot([0, np.real(wmax)], [0, np.imag(wmax)], 'm')
    plt.plot([0, np.real(wmin)], [0, np.imag(wmin)], 'm')
    plt.xlabel('u')
    plt.ylabel('v')
    plt.plot(np.real(a), np.imag(a), 'r')
    plt.plot(np.real(b), np.imag(b), 'g')
    plt.plot([0, np.real(a[0])], [0, np.imag(a[0])], 'ro')
    plt.plot([0, np.real(b[0])], [0, np.imag(b[0])], 'go')
    plt.plot([0, np.real(w[0])], [0, np.imag(w[0])], 'bo')
    plt.plot(np.real(a[0]), np.imag(a[0]), 'ro')
    plt.plot(np.real(b[0]), np.imag(b[0]), 'go')
    plt.plot(np.real(w[0]), np.imag(w[0]), 'bo')
    plt.plot(np.real([a[0], a[0]+b[0]]), np.imag([a[0], a[0]+b[0]]), linestyle='--', color='g')
    plt.plot(np.real([b[0], a[0]+b[0]]), np.imag([b[0], a[0]+b[0]]), linestyle='--', color='r')

    for n in range(len(ot)):
        plt.hold('on')
        plt.plot(np.real(a[n]), np.imag(a[n]), 'ro')
        plt.plot(np.real(b[n]), np.imag(b[n]), 'go')
        plt.plot(np.real(w[n]), np.imag(w[n]), 'bo')

    plt.hold('off')
    plt.show()


def prep_plot(SEMA, ECC, INC, PHA):
    """
    Take the output of ap2ep (SEMA, ECC, INC, and PHA) and prepare it for
    plotting.

    This is extracted from do_the_plot above, but allows quicker access when
    all that is required is a plot of an ellipse, for which only w is really
    required.

    Returns w, wmin and wmax (w is used for plotting the ellipse, see
    plot_ell).

    """

    i = 1j

    SEMI = SEMA * ECC
    Wp = (1 + ECC) / 2 * SEMA
    Wm = (1 - ECC) / 2 * SEMA
    THETAp = INC - PHA
    THETAm = INC + PHA

    # Convert degrees into radians
    THETAp = THETAp / 180 * np.pi
    THETAm = THETAm / 180 * np.pi
    INC = INC / 180 * np.pi
    PHA = PHA / 180 * np.pi

    # Calculate wp and wm.
    wp = Wp * np.exp(i * THETAp)
    wm = Wm * np.exp(i * THETAm)

    dot = np.pi / 36
    ot = np.arange(0, 2 * np.pi, dot)
    a = wp * np.exp(i * ot)
    b = wm * np.exp(-i * ot)
    w = a + b

    # Repeat the first position in w so we close the ellipse.
    w = np.hstack((w, w[0]))

    wmax = SEMA * np.exp(i * INC)
    wmin = SEMI * np.exp(i * (INC + np.pi / 2))

    return w, wmin, wmax


if __name__ == '__main__':

    """
    Replicate the tidal ellipse example file from Zhigang Xu's tidal_ellipse
    MATLAB toolbox.

    Pierre Cazenave (Plymouth Marine Laboratory), October 2012.

    """

    # Demonstrate how to use ap2ep and ep2ap
    Au = np.random.random([4, 3, 2])           # so 4x3x2 multi-dimensional matrices
    Av = np.random.random([4, 3, 2])           # are used for the demonstration.
    Phi_v = np.random.random([4, 3, 2]) * 360  # phase lags inputs are expected to
    Phi_u = np.random.random([4, 3, 2]) * 360  # be in degrees.

    plt.figure(1)
    plt.clf()
    SEMA, ECC, INC, PHA, w = ap2ep(Au, Phi_u, Av, Phi_v, [2, 3, 1])
    plt.figure(2)
    plt.clf()
    rAu, rPhi_u, rAv, rPhi_v, rw = ep2ap(SEMA, ECC, INC, PHA, [2, 3, 1])

    # Check if ep2ap has recovered Au, Phi_u, Av, Phi_v
    print(np.max(np.abs(rAu - Au).flatten()))        # = 9.9920e-16, = 2.22044604925e-16
    print(np.max(np.abs(rAv - Av).flatten()))        # = 6.6613e-16, = 7.77156117238e-16
    print(np.max(np.abs(rPhi_u - Phi_u).flatten()))  # = 4.4764e-13, = 1.70530256582e-13
    print(np.max(np.abs(rPhi_v - Phi_v).flatten()))  # = 1.1369e-13, = 2.27373675443e-13
    print(np.max(np.max(np.abs(w - rw).flatten())))  # = 1.3710e-15, = 1.1322097734e-15
    # For the random realization I (Zhigang Xu) had, the differences are listed
    # on the right hand of the above column. I (Pierre Cazenave) got the second
    # column with the Python version. What are yours?

    # Zhigang Xu
    # Nov. 12, 2000
    # Pierre Cazenave
    # October, 2012


def zoom_effect02(ax1, ax2, **kwargs):
	"""
	ax1 : the main axes
	ax1 : the zoomed axes

	Similar to zoom_effect01.  The xmin & xmax will be taken from the
	ax1.viewLim.
	"""

	import matplotlib.pyplot as plt
	import matplotlib.dates as dates
	from matplotlib.backends.backend_pdf import PdfPages

	from matplotlib.transforms import Bbox, TransformedBbox, \
	 blended_transform_factory

	from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector,\
	 BboxConnectorPatch

	tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
	trans = blended_transform_factory(ax2.transData, tt)

	mybbox1 = ax1.bbox
	mybbox2 = TransformedBbox(ax1.viewLim, trans)

	prop_patches=kwargs.copy()
	prop_patches["ec"]="none"
	prop_patches["alpha"]=0.2

	c1, c2, bbox_patch1, bbox_patch2, p = \
	connect_bbox(mybbox1, mybbox2,
	         loc1a=3, loc2a=2, loc1b=4, loc2b=1,
	         prop_lines=kwargs, prop_patches=prop_patches)

	ax1.add_patch(bbox_patch1)
	ax2.add_patch(bbox_patch2)
	ax2.add_patch(c1)
	ax2.add_patch(c2)
	ax2.add_patch(p)

	return c1, c2, bbox_patch1, bbox_patch2, p

def connect_bbox(bbox1, bbox2,
             loc1a, loc2a, loc1b, loc2b,
             prop_lines, prop_patches=None):

	import matplotlib.pyplot as plt
	import matplotlib.dates as dates
	from matplotlib.backends.backend_pdf import PdfPages

	from matplotlib.transforms import Bbox, TransformedBbox, \
	 blended_transform_factory

	from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector,\
	 BboxConnectorPatch

	if prop_patches is None:
		prop_patches = prop_lines.copy()
		prop_patches["alpha"] = prop_patches.get("alpha", 1)*0.2

	c1 = BboxConnector(bbox1, bbox2, loc1=loc1a, loc2=loc2a, **prop_lines)
	c1.set_clip_on(False)
	c2 = BboxConnector(bbox1, bbox2, loc1=loc1b, loc2=loc2b, **prop_lines)
	c2.set_clip_on(False)

	bbox_patch1 = BboxPatch(bbox1, **prop_patches)
	bbox_patch2 = BboxPatch(bbox2, **prop_patches)

	p = BboxConnectorPatch(bbox1, bbox2,
	                       #loc1a=3, loc2a=2, loc1b=4, loc2b=1,
	                       loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
	                       **prop_patches)
	p.set_clip_on(False)

	return c1, c2, bbox_patch1, bbox_patch2, p
