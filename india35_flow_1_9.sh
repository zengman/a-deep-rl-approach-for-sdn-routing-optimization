#!/bin/bash

nohup python3 Environment2.py 9 matTU2/mat9_2 >/dev/null 2>&1 &
nohup python3 Environment2.py 10 matTU2/mat10_2 >/dev/null 2>&1 &
nohup python3 Environment2.py 11 matTU2/mat11_2 >/dev/null 2>&1 &
nohup python3 Environment2.py 12 matTU2/mat12_2 >/dev/null 2>&1 &
nohup python3 Environment2.py 13 matTU2/mat13_2 >/dev/null 2>&1 &
nohup python3 Environment2.py 14 matTU2/mat14_2 >/dev/null 2>&1 &

nohup python DDPGfor.py 9 >/dev/null 2>&1 &
#nohup python DDPGfor.py 3 >/dev/null 2 &1 &
#nohup python DDPGfor.py 4 >/dev/null 2 &1 &
#nohup python DDPGfor.py 5 >/dev/null 2 &1 &
#nohup python DDPGfor.py 6 >/dev/null 2 &1 &
#nohup python DDPGfor.py 8 >/dev/null 2 &1 &
#nohup python DDPGfor.py 9 >/dev/null 2 &1 &
