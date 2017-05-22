import cv2
import os

class DebugImages:
    def __init__(self, dbgdir = None, show = False, in_row = None, screen_width = 1800):
        self.screen_width = screen_width
        self.top_border = 50
        self.fname = ""
        self.window_height = -1
        self.window_width = -1
        self.dbgdir = dbgdir
        self.show = show
        self.in_row = in_row
        self.active = True
        self.off_cnt = 0
        self.active = True
        self.frame_no = 0
        self.step_end(False)

    def set_active(self, active):
        if active:
            self.off_cnt -= 1
        else:
            self.off_cnt += 1
        self.active = (self.off_cnt == 0)

    def set_fname(self, fname, video = False):
        self.fname = fname
        self.is_video = video

    def __call__(self, stage, img):
        #print("CALL")
        self.dbg(stage, img)

    def dbg(self, stage, img):
        self.no+=1
        print("DBG1")
        if not self.active:
            return
        fname = self.fname
        if self.dbgdir is not None:
            bn =  os.path.basename(fname)
            i = bn.rfind(".")
            vid = ""
            if self.is_video:
                vid="_%04d_" % self.frame_no
            fname = self.dbgdir + "/" + bn[:i] + vid + ("_%02d_"%self.no) + stage + bn[i:]
            if img.dtype== 'float64':
                img = (img*255).astype('uint8')
            print("saving '%s' (%s)" % (fname, img.dtype))
            cv2.imwrite(fname, img)
        if self.show:
            if not self.in_row is None:
                fx = img.shape[1] / (float(self.screen_width)/self.in_row)
                if(fx > 1 or 1):
                    img = cv2.resize(img, (int(img.shape[1]/fx), int(img.shape[0]/fx)) )
            if self.window_height < 0:
                self.window_height = img.shape[0]
                self.window_width = img.shape[1] 
            x_off = self.x * self.window_width
            if x_off + self.window_width > self.screen_width:
                self.x = 0
                self.y += 1
            x_off = self.x * self.window_width + 100
            y_off = self.y * (self.window_height + self.top_border)  + 20

            self.x += 1
            #print("SHOW !!!")
            cv2.imshow(stage, img)
            cv2.moveWindow(stage, x_off, y_off)
 
    def step_end(self, wait = True, wtime = 0):
        self.x = 0
        self.y = 0
        self.no = 0
        self.frame_no += 1
        if self.show and wait:
            return cv2.waitKey(wtime) == ord('q')


def setup(dbgdir = None, show = False, in_row = None, screen_width = 1800):
    dbg.__init__(dbgdir, show, in_row, screen_width)

# do nothing default debug logger
dbg = DebugImages()
