cp -TR ./DataFiles/DefaultPatient "./Generated_Patients/patient_thrombolysis_example/"
python3 generate_files.py "./Generated_Patients/patient_thrombolysis_example/"

python3 run_simulation.py "./Generated_Patients/patient_thrombolysis_example/" 0
cp -TR  "./Generated_Patients/patient_thrombolysis_example/bf_sim/" "./Generated_Patients/patient_thrombolysis_example/Blood_flow_healthy"
python3 run_simulation.py "./Generated_Patients/patient_thrombolysis_example/" 1
cp -TR  "./Generated_Patients/patient_thrombolysis_example/bf_sim/" "./Generated_Patients/patient_thrombolysis_example/Blood_flow_stroke"

python3 ./Blood_Flow_1D/Thrombolysis1D.py "./Generated_Patients/patient_thrombolysis_example/"
cp -TR  "./Generated_Patients/patient_thrombolysis_example/bf_sim/" "./Generated_Patients/patient_thrombolysis_example/Blood_flow_thrombolysis"
