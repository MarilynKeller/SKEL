# Fitting SKEL to OpenSim

Here we explain how to fit the SKEL model to an OpenSim model sequence to get a surface mesh. We do not provide fitting code yet but if you are interested in this direction, this folder should give you a good starting point.

As an example, we will use the OpenSim model from https://simtk.org/projects/full_body.
Download the model and sample data from (this page)[https://simtk.org/projects/full_body]. You should get the zip folder: `FullBodyModelwSampleSim.-latest.zip` and unzip it. This folder contains a .osim OpenSim model and example .mot motion sequences.

You can visualize the sequence with: 

```bash
python examples/load_osim.py --osim /path/to/FullBodyModelwSampleSim.-latest/ModelWithSampleSimulations-4.0/SimulationDataAndSetupFiles-4.0/Rajagopal2015.osim --mot /path/to/FullBodyModelwSampleSim.-latest/ModelWithSampleSimulations-4.0/SimulationDataAndSetupFiles-4.0/IK/results_run/ik_output_run.mot
```

To fit SKEL to this sequence, we need to map each joint of this OpenSim model to a joint of the SKEL model. For the current model, the mapping is in `SKEL/skel/fit_osim/mappings/full_body.yaml`. If you use another model, you will need to create your own mapping yaml for that model.

The following script will then show you how to get the needed joints trajectories to fit SKEL to this OpenSim sequence:

```
python SKEL/skel/fit_osim/osim_fitter.py
```
