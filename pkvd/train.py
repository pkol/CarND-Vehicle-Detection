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
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
import pickle
import scipy.ndimage as ndi
from skimage import exposure

def fname2features(fname):
    img = cv2.imread(fname)
    #print(img.dtype, np.max(img), np.min(img))
    f = get_features(img)
    return f

def zoom_all(imgs):
    ret = []
    w,h = imgs[0].shape[0], imgs[0].shape[1]

    for img in imgs:
        f = np.random.random() +1
        i = cv2.resize(img, None,  fx=f, fy=f)
        max_x0 = i.shape[0] - w
        x0 = int(np.random.random() * max_x0)
        y0 = int(np.random.random() * max_x0)
        ret.append(i[x0:x0+w, y0:y0+w])
    return ret


def img2features(img):
    return CarFrame(img).get_features(0,0)

def get_xy(dir_name):
    imgs_veh    = [cv2.imread(fname) for fname in glob.glob(dir_name + '/vehicles/**/*.png')[::]]
    x = []
    x.extend( Parallel(n_jobs=4)(delayed(img2features)(img) for img in  imgs_veh))
    imgs_veh = None
    no_cars = len(x)
    imgs_nonveh = [cv2.imread(fname) for fname in glob.glob(dir_name + '/non-vehicles/**/*.png')[::]]
    x.extend( Parallel(n_jobs=4)(delayed(img2features)(img) for img in  imgs_nonveh))
    x.extend( Parallel(n_jobs=4)(delayed(img2features)(img) for img in  zoom_all(imgs_nonveh[::1])))
    imgs_nonveh = None
    y = np.zeros(len(x))
    y[:no_cars] = 1
    x = np.array(x)
    return (x,y)

def train_svm(x,y,C=1):
    #in_test = np.zeros(len(y))
    x_train, x_test, y_train, y_test = train_test_split( x,y, test_size = 0.05)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test  = scaler.transform(x_test)

    svc = LinearSVC(C=C)
    #svc = SVC(C=C)
    svc.fit(x_train,y_train)
    err_train = 1-np.sum(svc.predict(x_train) ==y_train) / len(x_train)
    err_test  = 1-np.sum(svc.predict(x_test) ==y_test) / len(x_test)
    print("Error on training set", err_train*100 , "%")
    print("Error on test set    ", err_test*100 , "%")
    return(svc, scaler, err_train*100, err_test*100)

def search_C(x,y):
    Cs = np.array(range(1,200))/100.0
    for c in Cs:
        print(c,*train_svm(x,y,c))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train car classifier')
    parser.add_argument('--data', nargs='?')
    parser.add_argument('-d', '--dbgdir', nargs='?')
    parser.add_argument('-s', '--show', help="show images", action='store_true')
    parser.add_argument('-m', '--model', help="filename for saved model" )
             
    args = parser.parse_args()
    debug.setup(args.dbgdir, args.show, in_row = 6)

    x,y = get_xy(args.data)
    print("Loaded")
    svc, scaler, e1,e2 = train_svm(x,y, 0.01)
    if args.model is not None:
        with open(args.model, 'wb') as f:
            pickle.dump((svc, scaler), f)


    #search_C(x,y)
