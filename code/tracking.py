import cv2 as cv
import numpy as np
from common import draw_str, RectSelector

def analyse (tracker, log):
    print ('---------------------------------------------')
    print ('Tracking/Detection/Classification finished for region label: %s ' % (str(tracker.label)))
    print ('Total number of detections: %d ' % (tracker.ndetections))
    print ('Number of frames of consecutive tracking: %d ' % (tracker.ntracking))
    #if tracker.ntracking > 0:
    #    print ('# detections / # tracking: %f ' % (tracker.ndetections/tracker.ntracking))
    # Modify the threshold below according to the expected minumum number of detections so as to avoid false alarms.
    # False alarms regions are rarely detected many times.
    if tracker.ndetections > 1: 
        if tracker.classification[0] > tracker.classification[1]:
           print ('The region %s is likely to be a mild imprint defect!' % (str(tracker.label)))
        else:    
           print ('The region %s is likely to be a severe imprint defect!' % (str(tracker.label)))
    print ('---------------------------------------------')

    log.write("---------------------------------------------\n")
    log.write ("Tracking/Detection/Classification for region label: %s\n" % (str(tracker.label)))
    log.write ('Total number of detections: %d\n' % (tracker.ndetections))
    log.write ('Number of frames of consecutive tracking: %d\n' % (tracker.ntracking))
    if tracker.ndetections > 1: 
        if tracker.classification[0] > tracker.classification[1]:
           log.write ('The region %s is likely to be a mild imprint defect!\n' % (str(tracker.label)))
        else:    
           log.write ('The region %s is likely to be a severe imprint defect!\n' % (str(tracker.label)))

'''
MOSSE tracking - This sample implements correlation-based tracking approach, described in [1].
[1] David S. Bolme et al. "Visual Object Tracking using Adaptive Correlation Filters"
    http://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf
'''

def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv.warpAffine(a, T, (w, h), borderMode = cv.BORDER_REFLECT)

def divSpec(A, B):
    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C

eps = 1e-5

class MOSSE:
    def __init__(self, frame, rect, ntrackers, nframe):
        x1, y1, x2, y2, dconf, dclas = rect
        w, h = map(cv.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h
        img = cv.getRectSubPix(frame, (w, h), (x, y))

        self.win = cv.createHanningWindow((w, h), cv.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv.dft(g, flags=cv.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        self.ndetections = 1
        self.ntracking = 1
        if dclas == 'P1':
           self.classification = [0,1]
        elif dclas == 'P2': 
           self.classification = [1,0]
        self.border = 0

        self.det_conf = dconf
        self.det_xmin = x1
        self.det_ymin = y1
        self.det_xmax = x2
        self.det_ymax = y2
        self.det_clas = dclas
        self.det_update = nframe

        self.label = ntrackers
        for _i in range(128):
            a = self.preprocess(rnd_warp(img))
            A = cv.dft(a, flags=cv.DFT_COMPLEX_OUTPUT)
            self.H1 += cv.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2 += cv.mulSpectrums(     A, A, 0, conjB=True)
        self.update_kernel()
        self.update(frame)

    def update(self, frame, rate = 0.125):
        (x, y), (w, h) = self.pos, self.size
        self.last_img = img = cv.getRectSubPix(frame, (w, h), (x, y))
        img = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        #self.good = self.psr > 8.0
        self.good = self.psr > 5.0
        if not self.good:
            return

        self.pos = x+dx, y+dy
        self.last_img = img = cv.getRectSubPix(frame, (w, h), self.pos)
        img = self.preprocess(img)

        A = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)
        H1 = cv.mulSpectrums(self.G, A, 0, conjB=True)
        H2 = cv.mulSpectrums(     A, A, 0, conjB=True)
        self.H1 = self.H1 * (1.0-rate) + H1 * rate
        self.H2 = self.H2 * (1.0-rate) + H2 * rate
        self.update_kernel()

    @property
    def state_vis(self):
        f = cv.idft(self.H, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT )
        h, w = f.shape
        f = np.roll(f, -h//2, 0)
        f = np.roll(f, -w//2, 1)
        kernel = np.uint8( (f-f.min()) / f.ptp()*255 )
        resp = self.last_resp
        resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)
        vis = np.hstack([self.last_img, kernel, resp])
        return vis

    def draw_state(self, vis, label):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        cv.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        if self.good:
            cv.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        else:
            cv.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
        draw_str(vis, (x1, y2+16), (('Region label: %s (track quality PSR: %.1f)') % (str(self.label), self.psr)))

    def preprocess(self, img):
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)
        return img*self.win

    def correlate(self, img):
        C = cv.mulSpectrums(cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv.idft(C, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
        h, w = resp.shape
        _, mval, _, (mx, my) = cv.minMaxLoc(resp)
        side_resp = resp.copy()
        cv.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)
        return resp, (mx-w//2, my-h//2), psr 

    def update_kernel(self):
        self.H = divSpec(self.H1, self.H2)
        self.H[...,1] *= -1

