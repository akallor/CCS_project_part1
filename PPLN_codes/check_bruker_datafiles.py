import pyTDFSDK
import numpy as np
import matplotlib.pyplot as plt

# Open the .d file
conn = pyTDFSDK.Connection()
conn.open("/content/CCS_project_part1/PPLN_codes/200705ko02phosphoChymoTryp_SCXFr00_inclALL_Slot1-01_1_11495.d.zip")

# Extract frame information
frames = conn.getFrameMsMsInfo()

# Extract 1/K0 values and m/z values
mz_values = []
invk0_values = []

for frame in frames:
    # Extract MS/MS spectra
    spectra = conn.getFrameMsMsSpectra(frame.Id)
    
    # Extract mobility values
    mobilogram = conn.getFrameMsMsMobilogram(frame.Id)
    
    # Get 1/K0 scale
    invk0_scale = conn.getInvIonMobilityCalibration(frame.Id)
    
    # Store values
    mz_values.append(frame.CollisionEnergy)
    invk0_values.append(invk0_scale)

# Plot distribution of 1/K0 values
plt.hist(np.concatenate(invk0_values), bins=50)
plt.xlabel("1/K0 (V·s/cm²)")
plt.ylabel("Count")
plt.title("Distribution of 1/K0 values")
plt.show()

conn.close()
