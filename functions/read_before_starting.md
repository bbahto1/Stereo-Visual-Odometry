## Processing the complete sequence of KITTI dataset
1) First of all check the python version you have on your device.
2) Your python version should be python 3.x
3) If you are using VSCODE on windows or linux operating system start the file with one of these commands
  - python3 SVO.py 0 1 800 1 or python SVO.py 0 1 800 1
4) First argument is the starting number of KITTI sequence, this can be changed accordingly (for more information I advise to look at KITTI datasets), 
second argument in this case is starting frame, third argument is the ending frame in the sequence 
(Disclaimer: If the sequence has less frames than the end frame, the processing will stop at the last frame of the sequence), 
the fourth argument is plotting live trajectory (if this argument is 1 than the live trajectory will be plotted, if this argument is 0 than live trajectory won't be plotted)
5) When plotting the live trajectory, when 100-th frame is processed the picture of the trajectory will be created, not before that.
