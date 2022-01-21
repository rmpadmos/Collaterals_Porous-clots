cp -TR ./DataFiles/DefaultPatient "./Generated_Patients/patient_collaterals_example/"
sed -i '26 c CollateralPatientGen=True' './Generated_Patients/patient_collaterals_example/bf_sim/Model_parameters.txt'
python3 generate_files.py "./Generated_Patients/patient_collaterals_example/"

python3 ./Blood_Flow_1D/CollateralsSimulation.py "./Generated_Patients/patient_collaterals_example/" 
python3 ./Blood_Flow_1D/ContrastModel.py "./Generated_Patients/patient_collaterals_example/" True 20
