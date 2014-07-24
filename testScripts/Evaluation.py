"""Author: Ankush Roy

  Input : trackerlines  -> File of tracked points
          groundtruthlines -> File of GT points
          TH -> Pixel threshold for succ
          t -> To simulate fast motion
"""

import math 
def main():
	frameRate = 1
	#trackerlines = open('/home/ankush/OriginalNN/Test/NNTracker/src/results/bookII/10000_1/nl_bookII_s3_10000.txt','r').readlines()
	trackerlines = open('/home/ankush/GNNTracker/Results/GNN/BookII_s3_4000_1/bookII_4000_1.txt','r').readlines()
	groundtruthlines = open("nl_bookII_s3.txt",'r').readlines()
	AnalysisData = []
	TH = 1
	while TH <= 16:
		Val = Success(trackerlines,groundtruthlines,TH,frameRate)
		print Val
		#print 'Succes for Threshold', TH, 'is', Val
		TH += 2

def Success(T,GT,TH,t):
	Error = []
	I = 1
	J = 1
	
	while J < len(T):
		#import pdb;pdb.set_trace()
		Tracker = T[J].strip().split()
		Groundtruth = GT[I].strip().split()
		Err = 0
		for i in range(1,9):
			Err  = (float(Tracker[i]) - float(Groundtruth[i]))**2 + Err
		if math.sqrt(Err/4) > TH:
			Error.append(0)
		else:
			Error.append(1)
		J = J + 1
		I = I + t # Step for fast motion
	return(sum(Error) / float(len(T) - 1))

if __name__ == '__main__':
	main()
