## Processing the complete sequence of KITTI dataset
1. First of all check the python version you have on your device.
1. Your python version should be python 3.x
1. If you are using VSCODE on windows or linux operating system start the file with one of these commands
   1. python3 SVO.py 0 1 800 1 or python SVO.py 0 1 800 1
1. Arguments: 
   1. First argument is the starting number of KITTI sequence, this can be changed accordingly (for more information I advise to look at KITTI datasets)
   1. Second argument in this case is starting frame, 
   1. Third argument is the ending frame in the sequence  (Disclaimer: If the sequence has less frames than the end frame, the processing will stop at the last frame of the sequence)
   1. The fourth argument is plotting live trajectory (if this argument is 1 than the live trajectory will be plotted, if this argument is 0 than live trajectory won't be plotted)
1. When plotting the live trajectory, when 100-th frame is processed the picture of the trajectory will be created, not before that.
