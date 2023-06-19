import numpy as np


def cut_spectrum(Dl, lrange):
    if lrange=='30_1999':
        Dl = Dl[30-2:2000-2] #30<=ell<=1999
    elif lrange=='2_1996':
        Dl = Dl[:1997-2] #2<=ell<=1996
    elif lrange=='2_1999':
        Dl = Dl[:2000-2] #2<=ell<=1999
    elif lrange=='2_2500':
        Dl = Dl[:2501-2] #2<=ell<=2500
    return Dl

def load_planck(component='tt', lrange='2_2500'):
    '''
    lrange: '30_1999', '2_1996', '2_1999' or '2_2500'
    component: 'tt', 'ee', 'te'
    '''
    d_ell = np.load('../data/planck_%s_all.npy'%component)
    d_ell = cut_spectrum(d_ell, lrange)
    return d_ell
