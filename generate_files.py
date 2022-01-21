#!user/bin/python3
import sys
from Blood_Flow_1D.GenerateBloodflowFiles import generatebloodflowfiles

assert len(sys.argv) == 2, "Requires a path"

# process patient data and generate files for blood flow simulations
generatebloodflowfiles(f"{sys.argv[1]}/")
