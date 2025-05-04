Welcome to my CSC420 Assignment 3 ReadMe!

Requirements:
 - python3.7
 - See imported libraries

Run command:
 - python3.7 <F_NAME>

---------------------------- 

NONE OF THE FIGURES ARE INCLUDED BUT CAN ALL BE EASILY GENERATED!
Most figures are in the report.

---------------------------- 

Task 1:
Entry point is hog.py, this is also the only file.

Running this will generate the plots in the assets subdir, as well as the txt. The 
txts are included but will be overwritten. All you need to do is change the img_name
variable in hog.py line 209. This will run it as needed. Now I would set up an arg parser
and give you the commands to make your life very easy but I think this will suffice. 
Run hog.py 4 times, each with a diffeerent name from: ('image1', 'image2', 'flash', 'noflash').

AKA:
for name in ('image1', 'image2', 'flash', 'noflash'):
    python hog.py (name)

---------------------------- 

Task 2:
You will need to run this two times, one for each value of sigma \in {0.5, 5} in line 81 
of main.py. But you must also change the threshold value. 

For sigma = 0.5:
    threshold_1 = 1000  
    threshold_2 = 7500  
For sigma = 5:
    threshold_1 = 2000  
    threshold_2 = 15000  
See the report in case this is wrong.
