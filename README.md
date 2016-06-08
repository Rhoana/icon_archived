Running icon:

1. open a terminal sessions listening on port 8889
   ssh -L 8889:localhost:8889 fgonda@rclogin09.rc.fas.harvard.edu

2. then request a compute node:
   srun --pty -p holyseasgpu --mem 90000 -t 18900 --tunnel 8889:8888 --gres=gpu:2 -n 4 -N 1 bash

3. run the web server
   cd rhoana/icon
   sh web.sh

4. run the trainining module
   cd rhoana/icon
   sh train.sh

5. run the segmentation module
   cd rhoana/icon
   sh segment.sh

6. from the web, you can chose a project and click start
   to activate it or stop to deactivate.

