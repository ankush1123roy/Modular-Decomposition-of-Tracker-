from Homography import *
from InteractiveTracking import *
from matplotlib import pyplot as plt
from matplotlib import animation
import os.path

class StandaloneTrackingApp(InteractiveTrackingApp):
    """ A demo program that uses OpenCV to grab frames. """

    def __init__(self, vc, init_frame, root_path,
                 params, labels, default_id, buffer_size):
        track_window_name = 'Tracked Images'
        InteractiveTrackingApp.__init__(self, init_frame, root_path, track_window_name,
                                        params, labels, default_id)
        self.vc = vc
        self.buffer_id=0
        self.buffer_end_id=-1
        self.buffer_start_id=0
        self.buffer_full=False
        self.buffer_size=buffer_size
        self.current_buffer_size=0
        self.frame_buffer=[]
        self.corners_buffer=[]
        self.actual_corners_buffer=[]
        self.rewind=False
        self.from_frame_buffer=False
        self.plot_index=0
        self.cam_skip_frames=50

    def run(self):
        i = self.init_frame
        while i < self.no_of_frames:
            img_path = self.img_path + '/img%d.jpg' % (i + 1)
            img = cv2.imread(img_path)
            if img == None:
                print("error loading image %s" % img_path)
                break
            if not self.on_frame(img, i):
                break
            i += 1
        self.cleanup()

    def updateTracker(self,i):
        if self.reset:
            self.reset=False
            self.initPlotParams()
        if self.exit_event or i==self.no_of_frames-1:
            return False
        if self.source=='camera':
            ret, img = self.cap.read()
            if not ret:
                print "Frame could not be read from camera"
                return False
            if not self.inited:
                print "Skipping ", self.cam_skip_frames, " frames...."
                for j in xrange(self.cam_skip_frames):
                    ret, img = self.cap.read()
        else:
            if i>=self.no_of_frames:
                self.exit_event=True
                return False
            img_file = self.img_path + '/img%d.jpg' % (i + 1)
            img = cv2.imread(img_file)
            if img == None:
                print "error loading image %s" % img_file
                return False

        if not self.on_frame(img, i):
            return False

        self.frame_times.append(i)
        self.curr_error_list.append(self.curr_error)
        self.avg_error=np.mean(self.curr_error_list)
        self.avg_error_list.append(self.avg_error)
        self.curr_fps_list.append(self.current_fps)
        self.avg_fps_list.append(self.average_fps)
        return True

    def onKeyPress(self, event):
        print 'key pressed=',event.key
        if event.key == "escape":
            self.exit_event=True
            sys.exit()
        elif event.key=="shift":
            if not self.from_cam or True:
                self.switch_plot=True
                self.plot_fps=not self.plot_fps
        elif event.key==" ":
            self.paused=not self.paused

    def keyboardHandler(self):
        key = cv2.waitKey(1)
        #if key!=-1:
        #    print "key=", key
        if key == ord(' '):
            self.paused = not self.paused
        elif key==27:
            return False
        elif key==ord('p') or key==ord('P'):
            if not self.from_cam or True:
                self.switch_plot=True
                self.plot_fps=not self.plot_fps
        elif key==ord('w') or key==ord('W'):
            self.write_res=not self.write_res
            if self.write_res:
                print "Writing results enabled"
            else:
                print "Writing results disabled"
        elif key==ord('r') or key==ord('R'):
            if self.from_frame_buffer:
                self.rewind=not self.rewind
                if self.rewind:
                    print "Disabling rewind"
                    #self.from_frame_buffer=False
                    #self.rewind=False
                else:
                    print "Enabling rewind"
                    #self.rewind=True
            else:
                print "Switching to frame buffer"
                print "Enabling rewind"
                self.from_frame_buffer=True
                self.rewind=True
                self.buffer_id=self.buffer_end_id
        return True


    def updatePlots(self, frame_count):
        if self.from_cam:
            ax.set_xlim(0, frame_count)
        if self.switch_plot:
            self.switch_plot=False
            print "here we are"
            if self.plot_fps:
                fig.canvas.set_window_title('FPS')
                plt.ylabel('FPS')
                plt.title('FPS')
                self.max_fps = max(self.curr_fps_list)
                ax.set_ylim(0, self.max_fps)
            else:
                fig.canvas.set_window_title('Tracking Error')
                plt.ylabel('Error')
                plt.title('Tracking Error')
                self.max_error = max(self.curr_error_list)
                ax.set_ylim(0, self.max_error)
            plt.draw()

        if self.plot_fps:
            line1.set_data(self.frame_times[0:self.plot_index+1], self.avg_fps_list[0:self.plot_index+1])
            line2.set_data(self.frame_times[0:self.plot_index+1], self.curr_fps_list[0:self.plot_index+1])
            #line3.set_data(self.frame_times[self.plot_index],self.curr_fps_list[self.plot_index])
            if max(self.curr_fps_list) > self.max_fps:
                self.max_fps = max(self.curr_fps_list)
                ax.set_ylim(0, self.max_fps)
                plt.draw()
        else:
            line1.set_data(self.frame_times[0:self.plot_index+1], self.avg_error_list[0:self.plot_index+1])
            line2.set_data(self.frame_times[0:self.plot_index+1], self.curr_error_list[0:self.plot_index+1])
            if max(self.curr_error_list)>self.max_error:
                self.max_error = max(self.curr_error_list)
                ax.set_ylim(0,self.max_error)
                plt.draw()

    def animate(self, i):
        if not self.keyboardHandler():
            sys.exit()

        if self.paused:
            return line1, line2

        if not self.buffer_full:
            if len(self.frame_buffer)>=self.buffer_size:
                print "Frame buffer full"
                #print "buffer_end_id=", self.buffer_end_id
                #print "buffer_start_id=", self.buffer_start_id
                self.buffer_full=True

        if self.from_frame_buffer:
            if self.rewind:
                self.buffer_id-=1
                self.plot_index-=1
                if self.buffer_id<0:
                    self.buffer_id=self.buffer_size-1
                elif self.buffer_id==self.buffer_start_id:
                    print "Disabling rewind"
                    self.rewind=False
            else:
                self.buffer_id+=1
                self.plot_index+=1
                if self.buffer_id>=self.buffer_size:
                    self.buffer_id=0
                elif self.buffer_id==self.buffer_end_id:
                    self.from_frame_buffer=False
                    print "Getting back to video stream"
            self.img=self.frame_buffer[self.buffer_id]
            self.corners=self.corners_buffer[self.buffer_id]
            self.actual_corners=self.actual_corners_buffer[self.buffer_id]
        else:
            self.plot_index=i
            if not self.updateTracker(i):
                self.writeResults()
                sys.exit()
            if not self.buffer_full:
                self.frame_buffer.append(self.img.copy())
                self.corners_buffer.append(self.corners.copy())
                self.actual_corners_buffer.append(self.actual_corners.copy())
                self.buffer_end_id+=1
            else:
                self.frame_buffer[self.buffer_start_id]=self.img.copy()
                self.corners_buffer[self.buffer_start_id]=self.corners.copy()
                self.actual_corners_buffer[self.buffer_start_id]=self.actual_corners.copy()
                self.buffer_end_id=self.buffer_start_id
                self.buffer_start_id=(self.buffer_start_id+1) % self.buffer_size

        if self.img != None:
            self.display()

        self.updatePlots(i)

        return line1, line2

def simData():
    i=-1
    while not app.exit_event:
        if not app.paused and not app.from_frame_buffer:
            i+=1
        if app.reset:
            print "Resetting the plots..."
            ax.cla()
            plt.draw()
            i=0
        yield i

if __name__ == '__main__':

    init_frame = 0
    frame_buffer_size=1000
    root_path = 'G:/UofA/Thesis/#Code/Datasets'

    sources=['file', 'camera']
    tracker_ids=['nn', 'esm', 'ict', 'l1']
    filter_ids=['none', 'gabor', 'laplacian', 'sobel', 'scharr', 'canny',
                'gauss', 'median', 'bilateral',
                'LoG', 'DoG']
    task_type=['simple', 'complex']
    actors=['Human', 'Robot']
    light_conditions=['nl', 'dl']
    speeds=['s1', 's2', 's3', 's4', 's5', 'si']
    complex_tasks=['bus', 'highlighting', 'letter', 'newspaper']
    simple_tasks=['bookI', 'bookII', 'bookIII', 'cereal', 'juice', 'mugI', 'mugII', 'mugIII']
    tasks=[simple_tasks, complex_tasks]
    params=[sources, filter_ids, tracker_ids, task_type, actors, light_conditions, speeds, tasks]
    labels=['source', 'filter', 'tracker', 'type', 'actor', 'light', 'speed', 'task']
    default_id=[0, 0, 3, 0, 0, 0, 2, 2]

    #nn_tracker = NNTracker(600, 2, res=(40, 40), use_scv=False)
    #esm_tracker = ESMTracker(5, res=(40, 40), use_scv=False)
    #ict_tracker=BakerMatthewsICTracker(10)
    #cascade_tracker = CascadeTracker([nn_tracker, esm_tracker])
    #
    #trackers=[nn_tracker, esm_tracker, ict_tracker, cascade_tracker]

    app = StandaloneTrackingApp(None, init_frame,root_path,
                                params, labels, default_id, frame_buffer_size)

    run_type=1
    if run_type==0:
        app.run()
    elif run_type==1:
        fig = plt.figure(0)
        fig.canvas.set_window_title('Tracking Error')
        cid = fig.canvas.mpl_connect('key_press_event', app.onKeyPress)
        ax = plt.axes(xlim=(0, app.no_of_frames), ylim=(0, 5))
        plt.xlabel('Frame')
        #plt.ylabel('Error')
        #plt.title('Tracking Error')
        plt.grid(True)
        line1, line2 = ax.plot([], [], 'r', [], [], 'g')
        plt.legend(('Average', 'Current'))
        #plt.draw()
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            #line3.set_data(0, 0)
            return line1,line2
        anim = animation.FuncAnimation(fig, app.animate, simData, init_func=init,
                                       interval=0,blit=True)
    #error = getTrackingError(dataset_path, res_path, dataset, tracker_id)
    plt.show()
    app.cleanup()

