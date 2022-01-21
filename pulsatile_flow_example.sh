# note: conv script does not seem to work
cp -TR ./DataFiles/DefaultPatient "./Generated_Patients/patient_pulsatile_example/"
sed -i '28 c Model=Pulsatile' './Generated_Patients/patient_pulsatile_example/bf_sim/Model_parameters.txt'
python3 generate_files.py "./Generated_Patients/patient_pulsatile_example/"

python3 ./Blood_Flow_1D/BloodFlowSimulator.py "./Pulsatile_Model/Blood Flow Model/bin/Debug/BloodflowModel.exe" "./Generated_Patients/patient_pulsatile_example/"
python3 ./Blood_Flow_1D/ContrastModel.py "./Generated_Patients/patient_pulsatile_example/" False 10 
cp -TR  "./Generated_Patients/patient_pulsatile_example/bf_sim/" "./Generated_Patients/patient_pulsatile_example/Blood_flow_healthy"

python3 ./Blood_Flow_1D/BloodFlowSimulator.py "./Pulsatile_Model/Blood Flow Model/bin/Debug/BloodflowModel.exe" "./Generated_Patients/patient_pulsatile_example/" --clot_present
python3 ./Blood_Flow_1D/ContrastModel.py "./Generated_Patients/patient_pulsatile_example/" False 10 
cp -TR  "./Generated_Patients/patient_pulsatile_example/bf_sim/" "./Generated_Patients/patient_pulsatile_example/Blood_flow_stroke"
