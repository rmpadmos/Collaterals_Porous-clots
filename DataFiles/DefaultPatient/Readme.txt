Original input files, per patient: 
-bf_sim/boundary_4&21&22&23&24&25&26&30.ply
-bf_sim/labelled_vol_mesh.msh
-bf_sim/Model_parameters.txt
-1-D_Anatomy.txt
-Clots.txt
-config.xml

Others:
-Optional patient segmentation or CoW file
-BraVa segmentation file
-MappingSystemVessels.vtp

Processed from boundary (see Remesh.py)
-PialSurface.vtp = remeshed version of boundary file
-PA = also remeshed version of boundary file with more triangles
-DualPA = Dualgraph of the PA file

