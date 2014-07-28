import curses

import os

import cv
import cv2
import numpy as np

from Metaio import *
from nntracker.utility import *
from nntracker.trackers.TurnkeyTrackers import *


base_dir = "/Users/travisdick/Desktop/metaio/"
video_dirs = []
os.path.walk(base_dir, lambda l,d,u: l.append(d + "/"), video_dirs)
video_dirs = video_dirs[1:]

algorithm_makers = {
    "ic":   lambda scv: make_bmic(res=(50,50), use_scv=scv),
    "esm":  lambda scv: make_esm(res=(50,50), use_scv=scv),
    "nnic": lambda scv: make_nn_bmic(res=(50,50), use_scv=scv)
}

results = {}
for key in algorithm_makers.iterkeys():
    results[key] = []

def main(win):
    for d in video_dirs:
        for n in xrange(1,6):
            for key in algorithm_makers.iterkeys():
                win.clear()
                win.addstr(0,0,"Sequence %d of %s" % (n, d))
                win.clrtoeol()
                win.addstr(1,0,"Algorithm: %s" % key)
                win.clrtoeol()
                win.refresh()
                results[key].append(load_and_run_benchmark(d, n, algorithm_makers[key](n==5)))
    
curses.wrapper(main)            
%
def make_video(outfile, results, names, colour, thickness):
    vw = cv2.VideoWriter(outfile, cv.CV_FOURCC('M','J','P','G'), 15, (640,480), 1)
    ri = 0
    for d in video_dirs:
        for n in xrange(1,6):
            vc, _ = open_benchmark(d, n)
            
            fi = 0
            while True:
                succ, img = vc.read()
                if not succ: break
                
                for name in names:
                    draw_region(img, results[name][ri][fi], colour[name], thickness[name])
                cv2.imshow("replay", img)
                cv2.waitKey(1)
                vw.write(img)
                fi += 1
                
            ri += 1
    
                
