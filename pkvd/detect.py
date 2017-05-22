import cv2
import glob
import numpy as np
import pickle
import argparse
import os
from debug import dbg
import debug
import imageio
from tqdm import tqdm
from pprint import pprint
from preproc import CarFrame
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle
import scipy.ndimage.measurements

def get_windows(img, wsize=64, wstep = 24):
    ret = []
    x=0
    while x + wsize < img.shape[1]:
        y=0
        while y + wsize < img.shape[0]:
            ret.append( (x,y ))
            y = y + wstep
        x = x + wstep
    return ret

def is_car(features, scv, scaler):
    feat = features.reshape(1,-1)
    feat = scaler.transform(feat)
    ret = svc.predict(feat)
    return ret[0]

def slide_window(img):
    wsize = 64
    cnt0 = 0
    cnt1 = 1
    out = img.copy()
    cf = CarFrame(img)
    ret = []
    for (x0,y0) in get_windows(img):
        patch = img[y0:y0+wsize, x0:x0+wsize]
        features = cf.get_features(x0,y0)
        res = is_car( features, svc, scaler)
        if res: 
            ret.append( (x0,y0, x0+wsize, y0+wsize) )
            #print(x0,y0)
            cv2.rectangle(out, (x0,y0), (x0+wsize, y0+wsize), (255,255,255))
            cnt1 += 1
        else:
            cnt0 += 1
        #img[y0:y0+wsize, x0:x0+wsize,0] += 50
    print(cnt0, cnt1)
    #dbg("detected %dx%d"%(img.shape[0], img.shape[1]), out)
    return(ret)

def plot_bb(img, bb, color, thick=1):
    for b in bb:
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color,thick)

def find_car(img):
    # touples: (scale, y_start, y_stop)
    cfg=[

            #(1.5,  0.5,  0.60),
            (1,    0.55,  0.65),
            (0.75, 0.55, 0.80), 
            (0.5,  0.55, 0.85), 
            ]
    
    i=0
    img_dbg = img.copy()
    hmap = np.zeros(img.shape[:-1], 'int')
    for (scale, y_start, y_stop) in cfg:
        i+=1
        y_start = int(y_start * img.shape[0])
        y_stop  = int(y_stop  * img.shape[0])
        cv2.line(img_dbg, (10+i*10, y_start), (10+i*10, y_stop), (200,100,100), 4)
    
        img_s = cv2.resize(img, None, fx=scale, fy=scale)
        img_s = img_s[int(y_start*scale):int(y_stop*scale)]
        bb = np.array(slide_window(img_s))
        if bb.shape[0]:
            bb = bb // (scale,scale, scale,scale)
            bb += (0,y_start, 0, y_start)
            bb = bb.astype('int')
            plot_bb(img_dbg, bb, (i*70 % 255, i*150 % 255, i*32 % 255))

        for b in bb:
            hmap[b[1]:b[3],b[0]:b[2]] += 1
    dbg("heat map", hmap.astype('uint8')*25)
    dbg("find_car", img_dbg)
    return hmap

class CarFinder:
    def __init__(self, hist_len):
        self.hist_len = hist_len
        self.hist = []

    def process_img(self, img):
        heat = find_car(img)
        self.hist.append(heat)
        if len(self.hist) > self.hist_len:
            self.hist.pop(0)
        h_sum = self.hist[0].copy()
        for i in range(1, len(self.hist)):
            h_sum += self.hist[i]
        return h_sum


def hmap2bbox(hmap, th):
    hmap[hmap < th] = 0
    la =  scipy.ndimage.measurements.label(hmap)
    dbg('label', la[0].astype('uint8')*50)
    ret = []
    for car_number in range(1, la[1]+1):
        nz = (la[0] == car_number).nonzero()
        nzy = nz[0]
        nzx = nz[1]
        bbox = (np.min(nzx), np.min(nzy), np.max(nzx), np.max(nzy))
        if (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) > 40*40: # filter out small boxes
            ret.append(bbox)
    return ret



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train car classifier')
    parser.add_argument('--data', nargs='?')
    parser.add_argument('-d', '--dbgdir', nargs='?')
    parser.add_argument('-s', '--show', help="show images", action='store_true')
    parser.add_argument('-m', '--model', help="filename for saved model" )
    parser.add_argument('-v', '--videoin' )
    parser.add_argument('-o', '--videoout' )
    parser.add_argument('-f', '--frame', type=int, default=0)
    parser.add_argument('images', help="test image", nargs='*' )
             
    args = parser.parse_args()
    debug.setup(args.dbgdir, args.show, in_row = 3)

    if args.model is not None:
        with open(args.model, 'rb') as f:
            svc,scaler = pickle.load( f)
    else: assert(0)
    for fname in args.images:
        img = cv2.imread(fname)
        print(img.dtype, np.max(img), np.min(img))
        dbg.set_fname(fname)
        dbg("input", img)
        print(dbg)
        find_car(img)

        dbg.step_end()

    if args.videoin:
        print("video ",args.videoin)
        #di.set_active(0)
        vid_reader = imageio.get_reader(args.videoin)
        fps = vid_reader.get_meta_data()['fps']

        cf = CarFinder(30)

        if args.videoout:
            vid_writer = imageio.get_writer(args.videoout, fps=fps)
        frm = 0
        for img in tqdm(vid_reader):
            dbg.set_fname(args.videoin+".png", video=True)
            frm += 1
            if(frm < args.frame): continue
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            dbg('input', img)
            hmap = cf.process_img(img)
            dbg('video hmap', hmap.astype('uint8'))
            for th in (60,):
                bbox = hmap2bbox(hmap, th)
                plot_bb(img, bbox, (50,200,th*4),th//10)
            dbg('img-bbox', img)
            if args.show:
                if dbg.step_end(wtime=5):
                    break
            if args.videoout:
                res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                vid_writer.append_data(res)

        if args.videoout:
            vid_writer.close()

       

