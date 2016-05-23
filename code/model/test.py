import os
import sys
import shutil

src_p = '/home/fgonda/icon/data/input/params/I00051.json'
dst_p = '/home/fgonda/icon/data/input/incoming/I00051.json'
lck_p = '/home/fgonda/icon/data/input/incoming/I00051.lock'

if os.path.exists( src_p ):
	shutil.move(src_p, dst_p)

if os.path.exists( lck_p ):
	os.remove(lck_p)
