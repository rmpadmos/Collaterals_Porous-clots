#!/usr/bin/python3
from desist.eventhandler.api import API
from desist.eventhandler.eventhandler import event_handler

from contextlib import suppress
import distutils.dir_util
import os
import pathlib
import shutil
import subprocess


def dict_to_xml(dictionary):
    import xml.etree.ElementTree as ET
    import xml.dom.minidom

    root = ET.Element("virtualPatient")
    patient = ET.SubElement(root, "Patient")

    # Directly convert each (key, value) into an XML element, except for the
    # events. These are given as a list and require specific treament.
    for key, val in dictionary.items():

        if key in 'models' or key == 'events' or key == 'labels':
            continue

        # directly convert the key to XML element with text set to its value
        el = ET.SubElement(patient, key)
        el.text = str(val)

    xml_str = ET.tostring(root, encoding="unicode")
    dom = xml.dom.minidom.parseString(xml_str)
    return dom.toprettyxml()


def update_patient_anatomy(patient, input_file, output_file):
    """Update the patient anatomy with patient-specific properties.

    The ``input_file`` refers to the original ``1-D_Anatomy.txt`` that is used
    as a template for the output. The relevant lines are modified and the new
    contents of the file are written to the path at ``output_file``.
    """
    ica_tag = "int. carotid"
    m1_tag = "MCA"
    delim = '\t'

    # The file is read in memory completely, these are sufficiently small that
    # we do not need to do any line-by-line type processing.
    with open(input_file, 'r') as infile:
        contents = [line for line in infile]

    # the modified file starts with the same header
    modified_lines = [contents[0]]

    ica_len = str(patient['ica_length_mm'])
    ica_rad = str(patient['ica_rad_mm'])
    m1_rad = str(patient['m1_rad_mm'])

    # Ignore the header and only changes lines that match the `ica_tag` and
    # `m1_tag` which indicate a line corresponds to the ICA or M1 vessel. For
    # these vessels patient-specific information is available and has to be
    # included within the anatomy description.
    for line in contents[1:]:
        if ica_tag in line:
            s = line.split(delim)
            line = delim.join([s[0], s[1], ica_len, ica_rad, ica_rad, s[-1]])

        if m1_tag in line:
            s = line.split(delim)
            line = delim.join([s[0], s[1], s[2], m1_rad, m1_rad, s[-1]])

        modified_lines.append(line)

    with open(output_file, 'w') as outfile:
        for line in modified_lines:
            outfile.write(line)


class API(API):
    def event(self):

        # Convert the `patient.yml` configuration to XML format as the
        # 1d-blood-flow reads its configuration information from XML only.
        # The file is converted on every call to ensure most up to date config.
        xml_config_path = self.patient_dir.joinpath('config.xml')

        # update the XML configuration file
        xml_config = dict_to_xml(dict(self.patient))
        with open(xml_config_path, 'w') as outfile:
            outfile.write(xml_config)

        # the subdirectory for blood-flow specific result files
        sim_dir = self.result_dir.joinpath("bf_sim/")
        os.makedirs(sim_dir, exist_ok=True)

        # copy the patient's configuration file and clots to result directory
        copy_files_from_to = [
            (self.patient_dir.joinpath('config.xml'), self.result_dir),
            (self.patient_dir.joinpath('Clots.txt'), self.result_dir),
        ]
        for source, destination in copy_files_from_to:
            shutil.copy(source, destination)

        # When running a subsequent evaluation of the blood flow model, e.g.
        # for the stroke or treatment phases, the original, initialised files
        # of the patient can be copied to the current result directory. This
        # acts as a "warm" start to the model and avoids reinitialising various
        # files for the blood flow model, which were already in place from the
        # original, "baseline" simulation from previous evaluations.
        if self.previous_event is not None:
            # FIXME: from `python 3.8` replace with `shutil.copytree()`,
            # current version does not support `shutil.copytree()` when some of
            # the directories already exist.
            distutils.dir_util.copy_tree(
                str(self.previous_result_dir),
                str(self.result_dir)
            )

        # We only perform the initialisation if the `Run.txt` file is not
        # present. This will only be the case on the first evaluation of this
        # model, as subsequent evaluations will copy this file to the current
        # simulation directory `sim_dir` already. Thus, changing any of the
        # values in the following files will influence _all_ steps.
        if not (os.path.isfile(sim_dir.joinpath("Run.txt"))):

            # The root directory is found with respect to the current file's
            # position on the file system, to ensure it is relative either
            # inside or outside of the container.
            root = pathlib.Path(os.path.dirname(__file__)).absolute()
            root = root.joinpath('DataFiles/DefaultPatient/')

            copy_default_files_from_to = [
                (root.joinpath('1-D_Anatomy.txt'), self.result_dir),
                (root.joinpath('bf_sim/labelled_vol_mesh.msh'), sim_dir),
                (root.joinpath('bf_sim/PialSurface.vtp'), sim_dir),
                (root.joinpath('bf_sim/Model_parameters.txt'), sim_dir),
            ]
            for source, destination in copy_default_files_from_to:
                # These files are only copied from the default patient
                # directory when they are *not* present on the expected
                # destination location. That should enable modification of such
                # parameter files by providing and alternative definition in
                # the expected location, which will then not be overwritten by
                # the defaults
                if not os.path.isfile(destination):
                    shutil.copy(source, destination)

            # Note: here we intervene with updating the patient anatomy. This
            # injects patient-specific anatomy information within the patient
            # anatomy definition. For instance, this updates the M1 and ICA
            # vessel radii.
            update_patient_anatomy(
                    self.patient,
                    self.result_dir.joinpath('1-D_Anatomy.txt'),
                    self.result_dir.joinpath('1-D_Anatomy_Patient.txt')
            )

            # This invokes the initialisation script to generate all require
            # patient files for the blood-flow simulation. Note, the
            # `1-D_Anatomy_Patient.txt` file is not updated anymore, as we
            # already generate this manually with patient-specific information
            # using the `update_patient_anatomy` function.
            subprocess.run([
                "python3",
                "generate_files.py",
                str(self.result_dir)
            ])

            # This cleans any large files that are require only during the
            # initialisation phase and not in any subsequent evaluations of
            # this model. This saves significant disk space.
            files_to_clean = [sim_dir.joinpath('Distancemat.npy')]
            for target in files_to_clean:
                with suppress(FileNotFoundError, IsADirectoryError) as _:
                    os.remove(target)

        # The file `clot_present` is emitted by the `place-clot` part of the
        # pipeline and indicates whether the clot should be considered in the
        # simulation. This is to distinguish between the various states of the
        # simulation, as the `Clots.txt` file containing all patient-specific
        # clot properties is present throughout any phase of the pipeline.
        clot_file = self.patient_dir.joinpath("clot_present")
        clot_flag = "0"

        if os.path.isfile(clot_file):
            clot_flag = "1"
            shutil.copy(str(clot_file), str(self.result_dir))

        print(f"Clot present: {clot_flag=}")

        # Finally, perform the actual simulation
        result = subprocess.run([
            "python3",
            "run_simulation.py",
            str(self.result_dir),
            clot_flag],
            stdout=subprocess.PIPE,
            check=True)

        print(result.stdout.decode('utf-8'))

    def example(self):
        self.event()

    def test(self):
        self.example()


if __name__ == "__main__":
    api = event_handler(API)
    api()
