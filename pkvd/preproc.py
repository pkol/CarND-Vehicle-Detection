import cv2
import numpy as np
from skimage.feature import hog
import argparse
import debug
from debug import  dbg

class CarFrame:
    def __init__(self, img, vis=False):
        self.img = img
        self.ppc = 8
        self.ws = 64 # windows size
        self.hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hls= self.hls
        p = {'block_norm':'L2-Hys', 'orientations':4, 'transform_sqrt':True, 'pixels_per_cell':(self.ppc,self.ppc), 'feature_vector':False}
        if vis:
            dbg("img H", hls[:,:,0])
            dbg("img L", hls[:,:,1])
            dbg("img S", hls[:,:,2])
            print(dbg)
            self.hog0,v = hog(hls[:,:,0], **p, visualise=True)
            dbg("HOG(H)",v)
            self.hog1,v = hog(hls[:,:,1], **p, visualise=True)
            dbg("HOG(L)",v)
            self.hog2,v = hog(hls[:,:,2], **p, visualise=True)
            dbg("HOG(S)",v)
        else:
            self.hog0 = hog(hls[:,:,0], **p)
            self.hog1 = hog(hls[:,:,1], **p)
            self.hog2 = hog(hls[:,:,2], **p)

    def get_hog(self, xi, yi):
        #xi, yi - x7y in the image. xh yh - in the self.hog
        xh = xi//8
        yh = yi//8
        h0 = self.hog0[yh:yh + 6, xh:xh+6].ravel()
        h1 = self.hog1[yh:yh + 6, xh:xh+6].ravel()
        h2 = self.hog2[yh:yh + 6, xh:xh+6].ravel()
        return np.concatenate([h1,h2])

    def get_features(self,xi,yi):
        nbins=32
        bins_range=(0,255)
        ws = self.ws
        #h0 = np.histogram(self.hls[xi:xi+ws, yi:yi+ws, 0], bins=nbins, range=bins_range)
        #h1 = np.histogram(self.hls[xi:xi+ws, yi:yi+ws, 1], bins=nbins, range=bins_range)
        #h2 = np.histogram(self.hls[xi:xi+ws, yi:yi+ws, 2], bins=nbins, range=bins_range)
        #f= np.concatenate([self.get_hog(xi,yi), h0[0], h1[0], h2[0]])
        f= np.concatenate([self.get_hog(xi,yi)])
        return f



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train car classifier')
    parser.add_argument('images', nargs = '+')
    parser.add_argument('-d', '--dbgdir', nargs='?')
    parser.add_argument('-s', '--show', help="show images", action='store_true')
             
    args = parser.parse_args()

    debug.setup(args.dbgdir, args.show, in_row = 6)
    print("GOT DBG")
    print(dbg)
    print(debug.dbg)

    print(args)
    for fname in args.images:
        dbg.set_fname(fname)
        img = cv2.imread(fname)
        dbg("input", img)
        CarFrame(img, True)

        #print("STEP END")
        if dbg.step_end():
            break

        


