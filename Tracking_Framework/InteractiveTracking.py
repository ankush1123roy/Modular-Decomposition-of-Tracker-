import os
import time
from TrackingParams import *
from FilteringParams import *
from Homography import *
from ImageUtils import *
from GUI import *
from Misc import *

from matplotlib import pyplot as plt

class InteractiveTrackingApp:
    def __init__(self, init_frame, root_path,
                 track_window_name, params, labels, default_id=None):

        self.root_path=root_path
        self.params=params
        self.labels=labels
        if default_id==None:
            default_id=[0 for i in xrange(len(self.params))]
        self.default_id=default_id
        self.first_call=True

        filter_index=labels.index('filter')
        self.filters_ids=dict(zip(params[filter_index], [i for i in xrange(len(params[filter_index]))]))
        self.filter_type=default_id[filter_index]
        self.filters_name=params[filter_index][self.filter_type]
        self.filters=[]
        for i in xrange(1, len(params[filter_index])):
            self.filters.append(FilterParams(params[filter_index][i]))

        tracker_index=labels.index('tracker')
        self.tracker_ids=dict(zip(params[tracker_index], [i for i in xrange(len(params[tracker_index]))]))
        self.tracker_type=default_id[tracker_index]
        self.trackers=[]
        for i in xrange(len(params[tracker_index])):
            self.trackers.append(TrackingParams(params[tracker_index][i]))

        self.source=default_id[labels.index('source')]

        self.init_frame=init_frame
        self.track_window_name = track_window_name
        self.proc_window_name='Processed Images'

        self.gray_img = None
        self.proc_img = None
        self.paused = False
        self.enable_blurring=False
        self.window_inited=False
        self.init_track_window=True
        self.img = None
        self.init_params=[]
        self.times = 1

        self.reset=False
        self.exit_event=False
        self.write_res=False
        self.cap=None

        self.initPlotParams()

        gui_title="Choose Input Video Parameters"
        self.gui_obj=GUI(self, gui_title)
        #self.gui_obj.initGUI()
        #self.gui_obj.root.mainloop()

    def initCamera(self):
        print "Getting input from camera"
        if self.cap!=None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        if self.cap==None:
            print "Could not access video camera"
            self.exit_event=True
            sys.exit()
        dWidth = self.cap.get(3)
        dHeight = self.cap.get(4)
        print "Frame size : ", dWidth, " x ", dHeight
        self.res_file = open('camera_res_%s.txt' % self.tracker_name, 'w')
        self.res_file.write('%-8s%-8s%-8s%-8s%-8s%-8s%-8s%-8s%-8s\n' % (
            'frame', 'ulx', 'uly', 'urx', 'ury', 'lrx', 'lry', 'llx', 'lly'))
        self.no_of_frames=0
        #sys.exit()

    def initVideoFile(self):
        type=self.init_params[self.labels.index('type')]
        actor=self.init_params[self.labels.index('actor')]
        light=self.init_params[self.labels.index('light')]
        speed=self.init_params[self.labels.index('speed')]
        task=self.init_params[self.labels.index('task')]

        self.dataset_path=self.root_path+'/'+actor
        self.res_path = self.dataset_path + '/results'
        if type=='simple':
            data_file = light+'_'+task+'_'+speed
        elif type=='complex':
            data_file = light+'_'+task
        else:
            print "Invalid task type specified: %s"%type
            return False

        self.data_file=data_file
        print "Getting input from data: ", self.data_file
        if not os.path.exists(self.res_path):
            os.mkdir(self.res_path)
        self.res_file = open(self.res_path + '/' + data_file + '_res_%s.txt'%self.tracker_name, 'w')
        self.res_file.write('%-8s%-8s%-8s%-8s%-8s%-8s%-8s%-8s%-8s\n' % (
            'frame', 'ulx', 'uly', 'urx', 'ury', 'lrx', 'lry', 'llx', 'lly'))

        self.img_path = self.dataset_path + '/' + data_file
        if not os.path.isdir(self.img_path):
            print 'Data directory does not exist: ', self.img_path
            self.exit_event=True
            return False
        self.ground_truth = readTrackingData(self.dataset_path + '/' + data_file + '.txt')
        self.no_of_frames = self.ground_truth.shape[0]
        print "no_of_frames=", self.no_of_frames
        self.initparam = [self.ground_truth[self.init_frame, 0:2].tolist(),
                     self.ground_truth[self.init_frame, 2:4].tolist(),
                     self.ground_truth[self.init_frame, 4:6].tolist(),
                     self.ground_truth[self.init_frame, 6:8].tolist()]
            #print tracking_data
        print "object location initialized to:",  self.initparam


    def initSystem(self, init_params):

        print "\n"+"*"*60+"\n"

        self.inited=False
        self.initFilterWindow()
        self.init_params=init_params

        self.source=init_params[self.labels.index('source')]

        self.tracker_name=init_params[self.labels.index('tracker')]
        self.tracker_type=self.tracker_ids[self.tracker_name]
        self.tracker=self.trackers[self.tracker_type].update()

        self.filters_name=init_params[self.labels.index('filter')]
        old_filter_type=self.filter_type
        self.filter_type=self.filters_ids[self.filters_name]
        if old_filter_type!=self.filter_type:
            self.initFilterWindow()

        if self.filter_type==0:
            print "Filtering disabled"
        elif self.filter_type <=len(self.filters):
            print "Using %s filtering" % self.filters[self.filter_type-1].type
        else:
            print 'Invalid filter type: ', self.filter_type
            return False

        #print "Using ", self.tracker_name, " tracker"

        if self.source=='camera':
            print "Initializing camera..."
            self.from_cam=True
            self.initCamera()
            self.plot_fps=True
        else:
            self.from_cam=False
            self.initVideoFile()
            self.plot_fps=False

        if not self.first_call:
            self.writeResults()

        print "\n"+"*"*60+"\n"
        return True

    def initPlotParams(self):

        self.curr_error=0
        self.avg_error=0
        self.avg_error_list=[]
        self.curr_fps_list=[]
        self.avg_fps_list=[]
        self.curr_error_list=[]
        self.frame_times=[]
        self.max_error=0
        self.max_fps=0
        self.max_val=0
        self.call_count=0

        self.count=0
        self.current_fps=0
        self.average_fps=0

        #self.start_time=datetime.now().time()
        self.start_time=0
        self.current_time=0
        self.last_time=0

        self.switch_plot=True

    def getTrackingObject(self):
        annotated_img=self.img.copy()
        temp_img=self.img.copy()
        title='Select the object to track'
        cv2.namedWindow(title)
        cv2.imshow(title, annotated_img)
        pts=[]

        def drawLines(img, hover_pt=None):
            if len(pts)==0:
                return
            for i in xrange(len(pts)-1):
                cv2.line(img, pts[i], pts[i+1], (0, 0, 255), 1)
            if hover_pt==None:
                return
            cv2.line(img, pts[-1], hover_pt, (0, 0, 255), 1)
            if len(pts)==3:
                cv2.line(img, pts[0], hover_pt, (0, 0, 255), 1)
            cv2.imshow(title, img)

        def mouseHandler(event, x, y, flags=None, param=None):
            if event==cv2.EVENT_LBUTTONDOWN:
                pts.append((x, y))
                drawLines(annotated_img)
            elif event==cv2.EVENT_LBUTTONUP:
                pass
            elif event==cv2.EVENT_RBUTTONDOWN:
                pass
            elif event==cv2.EVENT_RBUTTONUP:
                pass
            elif event==cv2.EVENT_MBUTTONDOWN:
                pass
            elif event==cv2. EVENT_MOUSEMOVE:
                if len(pts)==0:
                    return
                temp_img=annotated_img.copy()
                drawLines(temp_img, (x,y))

        cv2.setMouseCallback(title, mouseHandler, param=[annotated_img, temp_img, pts])
        while len(pts)<4:
            key=cv2.waitKey(1)
            if key==27:
                break
        cv2.destroyWindow(title)
        cv2.waitKey(1500)
        return pts

    def on_frame(self, img, numtimes):
        #print "frame: ", numtimes
        if self.first_call:
            self.gui_obj.initWidgets(start_label='Reset')
            self.first_call=False

        self.count+=1
        self.times = numtimes

        self.img = img
        self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.gray_img_float = self.gray_img.astype(np.float64)

        self.proc_img=self.applyFiltering()
        cv2.imshow(self.proc_window_name,  self.proc_img)

        self.proc_img = self.proc_img.astype(np.float64)

        if not self.inited:
            cv2.namedWindow(self.track_window_name)
            if self.from_cam:
                pts=self.getTrackingObject()
                if len(pts)<4:
                    self.exit_event=True
                    sys.exit()
                init_array = np.array(pts).T
            else:
                init_array = np.array(self.initparam).T

            self.tracker.initialize(self.proc_img, init_array)

            self.start_time=time.clock()
            self.current_time=self.start_time
            self.last_time=self.start_time

            self.inited = True

        self.tracker.update(self.proc_img)
        self.corners = self.tracker.get_region()

        if not self.from_cam:
            self.actual_corners = [self.ground_truth[self.times - 1, 0:2].tolist(),
                              self.ground_truth[self.times - 1, 2:4].tolist(),
                              self.ground_truth[self.times - 1, 4:6].tolist(),
                              self.ground_truth[self.times - 1, 6:8].tolist()]
            self.actual_corners=np.array(self.actual_corners).T
        else:
            self.actual_corners=self.corners.copy()

        self.updateError(self.actual_corners, self.corners)

        return True

    def updateError(self, actual_corners, tracked_corners):
        self.error=0
        if self.from_cam:
            self.curr_error=0
            return
        for i in xrange(2):
            for j in xrange(4):
                self.curr_error += math.pow(actual_corners[i, j] - tracked_corners[i, j], 2)
        self.curr_error = math.sqrt(self.curr_error / 4)

    def display(self):
        annotated_img = self.img.copy()
        if self.tracker.is_initialized():
            draw_region(annotated_img, self.corners, (0, 0, 255), 2)
            draw_region(annotated_img, self.actual_corners, (0, 255, 0), 2)
            self.res_file.write('%-15s%-12.2f%-12.2f%-12.2f%-12.2f%-12.2f%-12.2f%-12.2f%-12.2f\n' % (
                'frame' + ('%05d' % (self.times)) + '.jpg', self.corners[0, 0],
                self.corners[1, 0], self.corners[0, 1], self.corners[1, 1],
                self.corners[0, 2], self.corners[1, 2], self.corners[0, 3],
                self.corners[1, 3]))

        self.last_time=self.current_time
        self.current_time=time.clock()

        self.average_fps=(self.times+1)/(self.current_time-self.start_time)
        self.current_fps = 1.0 / (self.current_time - self.last_time)

        fps_text = "%5.2f"%self.average_fps + "   %5.2f"%self.current_fps
        cv2.putText(annotated_img, fps_text, (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (255,255,255))
        cv2.imshow(self.track_window_name, annotated_img)

    def applyFiltering(self):
        if self.filter_type == 0:
            proc_img = self.gray_img
        elif self.filter_type == self.filters_ids['DoG'] or \
                        self.filter_type == self.filters_ids['gauss'] or \
                        self.filter_type == self.filters_ids['bilateral'] or \
                        self.filter_type == self.filters_ids['median'] or \
                        self.filter_type == self.filters_ids['canny']:
             proc_img=self.filters[self.filter_type-1].apply(self.gray_img)
        elif self.filter_type <=len(self.filters):
            proc_img=self.filters[self.filter_type-1].apply(self.gray_img_float)
        else:
            print "Invalid filter type"
            return None
        return proc_img

    def initFilterWindow(self):
        if self.window_inited:
            cv2.destroyWindow(self.proc_window_name)
            self.window_inited=False
        cv2.namedWindow(self.proc_window_name,flags=cv2.CV_WINDOW_AUTOSIZE)
        if self.filter_type>0:
            for i in xrange(len(self.filters[self.filter_type-1].params)):
                cv2.createTrackbar(self.filters[self.filter_type-1].params[i].name, self.proc_window_name, self.filters[self.filter_type-1].params[i].multiplier,
                                   self.filters[self.filter_type-1].params[i].limit, self.updateFilterParams)
        self.window_inited=True

    def updateFilterParams(self, val):
        if self.filters[self.filter_type-1].validated:
            return
        #print 'starting updateFilterParams'
        for i in xrange(len(self.filters[self.filter_type-1].params)):
            new_val=cv2.getTrackbarPos(self.filters[self.filter_type-1].params[i].name, self.proc_window_name)
            old_val=self.filters[self.filter_type-1].params[i].multiplier
            if new_val!=old_val:
                self.filters[self.filter_type-1].params[i].updateValue(new_val)
                if not self.filters[self.filter_type-1].validate():
                    self.filters[self.filter_type-1].params[i].updateValue(old_val)
                    cv2.setTrackbarPos(self.filters[self.filter_type-1].params[i].name, self.proc_window_name,
                                       self.filters[self.filter_type-1].params[i].multiplier)
                    self.filters[self.filter_type-1].validated=False
                break
        self.filters[self.filter_type-1].kernel = self.filters[self.filter_type-1].update()
        if self.write_res:
            self.write_res=False
            self.writeResults()
        self.reset=True

    def getParamStrings(self):
        dataset_params=''
        if self.from_cam:
            dataset_params='cam'
        else:
            for i in xrange(1, len(self.init_params)):
                dataset_params=dataset_params+'_'+self.init_params[i]
            dataset_params=dataset_params+'_%d'%self.times
        filter_id='none'
        filter_params=''
        if self.filter_type>0:
            filter_id=self.filters[self.filter_type-1].type
            for i in xrange(len(self.filters[self.filter_type-1].params)):
                filter_params=filter_params+'_'+self.filters[self.filter_type-1].params[i].name\
                              +'_%d'%self.filters[self.filter_type-1].params[i].val
        tracker_params=''
        tracker_id=self.trackers[self.tracker_type-1].type
        for i in xrange(len(self.trackers[self.tracker_type-1].params)):
            tracker_params=tracker_params+'_'+self.trackers[self.tracker_type-1].params[i].name\
                          +'_%d'%self.trackers[self.tracker_type-1].params[i].val
        return [dataset_params, filter_id, filter_params, tracker_params]

    def writeResults(self):
        print('Saving results...')
        [dataset_params, filter_id, filter_params, tracking_params]=self.getParamStrings()
        self.max_fps = max(self.curr_fps_list[1:])
        min_fps=min(self.curr_fps_list[1:])
        self.max_error = max(self.curr_error_list)

        tracking_res_fname='Results/summary.txt'
        if not os.path.exists(tracking_res_fname):
            res_file=open(tracking_res_fname, 'a')
            res_file.write("tracker".ljust(10)+
            "\tfilter".ljust(10)+
            "\tavg_error".rjust(10)+
            "\tmax_error".rjust(10)+
            "\tavg_fps".rjust(10)+
            "\tmax_fps".rjust(10)+
            "\tmin_fps".rjust(10)+
            "\tdataset".center(50)+
            "\tfilter params".center(50)+
            "\ttracking params".center(50)+'\n')
        else:
            res_file=open(tracking_res_fname, 'a')

        res_file.write(self.tracker_name.ljust(10)+
                       '\t'+filter_id.ljust(10)+
                       '\t%10.6f'%self.avg_error+
                       '\t%10.6f'%self.max_error+
                       '\t%10.6f'%self.average_fps+
                       '\t%10.6f'%self.max_fps+
                       '\t%10.6f'%min_fps+
                       '\t'+dataset_params.center(50)+
                       '\t'+filter_params.center(50)+
                       '\t'+tracking_params.center(50)+'\n')
        res_file.close()
        self.savePlots(dataset_params, filter_id, filter_params)

    def generateCombinedPlots(self):
        combined_fig=plt.figure(1)
        plt.subplot(211)
        plt.title('Tracking Error')
        plt.ylabel('Error')
        plt.plot(self.frame_times, self.avg_error_list, 'r',
                 self.frame_times, self.curr_error_list, 'g')

        plt.subplot(212)
        plt.title('FPS')
        plt.xlabel('Frame')
        plt.ylabel('FPS')
        plt.plot(self.frame_times, self.avg_fps_list, 'r',
                 self.frame_times, self.curr_fps_list, 'g')
        return combined_fig

    def savePlots(self,dataset_params, filter_id, filter_params):
        print('Saving plot data...')

        plot_dir="Results/"+self.tracker_name+'/'+filter_id
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        plot_fname=plot_dir+'/'+dataset_params+'_'+filter_params

        combined_fig=self.generateCombinedPlots()

        combined_fig.savefig(plot_fname, ext='png', bbox_inches='tight')
        plt.figure(0)

        res_fname=plot_fname+'.txt'
        res_file=open(res_fname,'w')
        res_file.write("curr_fps".rjust(10)+"\t"+"avg_fps".rjust(10)+"\t\t"+
                       "curr_error".rjust(10)+"\t"+"avg_error".rjust(10)+"\n")
        for i in xrange(len(self.avg_fps_list)):
            res_file.write("%10.5f\t" % self.curr_fps_list[i] +
                           "%10.5f\t\t" % self.avg_fps_list[i] +
                           "%10.5f\t" % self.curr_error_list[i] +
                           "%10.5f\n" % self.avg_error_list[i])
        #print "done"

    def cleanup(self):
        self.res_file.close()

