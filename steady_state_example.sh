cp -TR ./DataFiles/DefaultPatient "./Generated_Patients/patient_steady_example/"
python3 generate_files.py "./Generated_Patients/patient_steady_example/"
python3 ./Blood_Flow_1D/vessel_mapping.py "./Generated_Patients/patient_steady_example/"

python3 run_simulation.py "./Generated_Patients/patient_steady_example/" 0
python3 ./Blood_Flow_1D/ContrastModel.py "./Generated_Patients/patient_steady_example/" True 10
cp -TR  "./Generated_Patients/patient_steady_example/bf_sim/" "./Generated_Patients/patient_steady_example/Blood_flow_healthy"

python3 run_simulation.py "./Generated_Patients/patient_steady_example/" 1
python3 ./Blood_Flow_1D/ContrastModel.py "./Generated_Patients/patient_steady_example/" True 10
cp -TR  "./Generated_Patients/patient_steady_example/bf_sim/" "./Generated_Patients/patient_steady_example/Blood_flow_stroke"


