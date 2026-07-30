# [[file:../../../../../../data/Nextcloud-SIG/dschmidt_working_projects/PC612_PS_RRM/PC612_emulator_v2.org::*Generate SD data][Generate SD data:1]]
import numpy as np
import rasterio as rio

BASE = "/mnt/share/rem"
PROJ = "PC612_emulator_WC711_v1"
OUTPUT_DIR_NAME = "ignition_location_uncertainty"

num_iterations = 30 # number of reps in which ign location can vary (min reps req'd for CLT)

PS_SCENARIO = "s2"
PS_DATA_DIR = f"{BASE}/{PROJ}/PS_scenarios"
PS_OUTPUT_DIR = f"{PS_DATA_DIR}/{PS_SCENARIO}/{OUTPUT_DIR_NAME}"
PS_proj_ids = np.arange(1, 11) # s2 (could read from gpkg)

template_raster = rio.open(f"{PS_OUTPUT_DIR}/iteration_0/1_baseline/1_bp.tif")
template_profile = template_raster.profile

for proj_id in PS_proj_ids:
    print(f"{proj_id=}")

    rd_cbp_arrays = []

    for iteration in range(num_iterations):
        baseline_cbp_a = rio.open(f"{PS_OUTPUT_DIR}/iteration_{iteration}/{proj_id}_baseline/{proj_id}_bp.tif").read(1)
        treatment_cbp_a = rio.open(f"{PS_OUTPUT_DIR}/iteration_{iteration}/{proj_id}_legalmax/{proj_id}_bp.tif").read(1)

        # mask for RD CBP calc and then set to max obs. value
        masked_baseline_cbp_a = np.ma.masked_where(baseline_cbp_a == 0, baseline_cbp_a)
        rd_cbp_a = (treatment_cbp_a - masked_baseline_cbp_a) / masked_baseline_cbp_a
        rd_cbp_a[(baseline_cbp_a == 0) & (treatment_cbp_a > 0)] = np.max(rd_cbp_a)
        #rd_cbp_a = np.ma.masked_where(rd_cbp_a == 0, rd_cbp_a) # only needed when symbolizing
        rd_cbp_arrays.append(rd_cbp_a)

    rd_cbp_std_a = np.std(rd_cbp_arrays, axis=0)
    rd_cbp_mean_a = np.average(rd_cbp_arrays, axis=0)
    print(f"{np.average(rd_cbp_std_a)=}, {np.average(rd_cbp_mean_a)=}") # average over the rep-averaged rasters

    with rio.open(f"{PS_OUTPUT_DIR}/pa{proj_id}_rd_cbp_std.tif", "w", **template_profile) as dst:
        dst.write(rd_cbp_std_a.astype(rio.float32), 1)

    with rio.open(f"{PS_OUTPUT_DIR}/pa{proj_id}_rd_cbp_mean.tif", "w", **template_profile) as dst:
        dst.write(rd_cbp_mean_a.astype(rio.float32), 1)
# Generate SD data:1 ends here
