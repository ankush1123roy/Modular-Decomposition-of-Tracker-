__author__ = 'Tommy'
import os
import sys
import numpy as np

def readTrackingData(filename):
    if not os.path.isfile(filename):
        print "Tracking data file not found:\n ",filename
        sys.exit()

    data_file = open(filename, 'r')
    data_file.readline()
    lines = data_file.readlines()
    no_of_lines = len(lines)
    data_array = np.empty([no_of_lines, 8])
    line_id = 0
    for line in lines:
        #print(line)
        words = line.split()
        if (len(words) != 9):
            msg = "Invalid formatting on line %d" % line_id + " in file %s" % filename + ":\n%s" % line
            raise SyntaxError(msg)
        words = words[1:]
        coordinates = []
        for word in words:
            coordinates.append(float(word))
        data_array[line_id, :] = coordinates
        #print words
        line_id += 1
    data_file.close()
    return data_array

def getTrackingError(ground_truth_path, result_path, dataset, tracker_id):
    ground_truth_filename = ground_truth_path + '/' + dataset + '.txt'
    ground_truth_data = readTrackingData(ground_truth_filename)
    result_filename = result_path + '/' + dataset + '_res_%s.txt' % tracker_id

    result_data = readTrackingData(result_filename)
    [no_of_frames, no_of_pts] = ground_truth_data.shape
    error = np.zeros([no_of_frames, 1])
    print "no_of_frames=", no_of_frames
    print "no_of_pts=", no_of_pts
    if result_data.shape[0] != no_of_frames or result_data.shape[1] != no_of_pts:
        print "no_of_frames 2=", result_data.shape[0]
        print "no_of_pts 2=", result_data.shape[1]
        raise SyntaxError("Mismatch between ground truth and tracking result")

    error_filename = result_path + '/' + dataset + '_res_%s_error.txt' % tracker_id
    error_file = open(error_filename, 'w')
    for i in xrange(no_of_frames):
        data1 = ground_truth_data[i, :]
        data2 = result_data[i, :]
        for j in xrange(no_of_pts):
            error[i] += math.pow(data1[j] - data2[j], 2)
        error[i] = math.sqrt(error[i] / 4)
        error_file.write("%f\n" % error[i])
    error_file.close()
    return error