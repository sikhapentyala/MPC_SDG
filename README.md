# CaPS: Collaborative and Private Synthetic Data Generation from Distributed Sources
We introduced a general framework CaPS that enables collaborative and private generation of tabular synthetic data based on real data from multiple data holders. CaPS follows the select-measure-generate template to generate syntheticdata regardless of how the real data is distibuted among the
data holders, i.e. horizontally, vertically, or mixed. We leverage the idea of “DP-in-MPC” to provide input privacy by performing computations on secret shared (i.e. encrypted) data, and applying differential privacy mechanisms within MPC itself. Letting MPC servers emulate a trusted central entity allows us to provide the same level of output 
privacy and utility as in the centralized paradigm, however without having to sacrifice input privacy. We demonstrated the applicability of CaPS for the state-of-the-art marginal based synthetic data generators AIM and MWEM+PGM.

![Screenshot](CaPS.pdf)

# Setup
- Install and setup private-pgm (https://github.com/ryan112358/private-pgm)
- Install and setup MP-SPDZ (https://github.com/data61/MP-SPDZ)
- Download the data in private-pgm/data/
- The code used for experiments is available in this_repo/private-pgm/mechanisms/<method>_MPC_<partition>.py

# Run the code
- Go to this_repo/private-pgm/mechanisms/
- Set the parameters and check for compilations
- Run the python file
Example: python3 aim_MPC_H.py for horizontal distributed setting
