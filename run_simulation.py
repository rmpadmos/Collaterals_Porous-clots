#!user/bin/python3
import sys
from Blood_Flow_1D.BloodFlowSimulator import blood_flow_script

assert len(sys.argv) == 3, "Requires two arguments."

executable = "/app/Pulsatile_Model/Blood Flow Model/bin/Debug/BloodflowModel.exe"
clot_present = True if sys.argv[2] == "1" else False

print("Running blood flow model. Storing output in: `{}`".format(sys.argv[1]))
print("Considering clot_present: `{}`.".format(clot_present))
print("Running executable: `{}`.".format(executable))

blood_flow_script(executable, "{}/".format(sys.argv[1]), clot_present)
