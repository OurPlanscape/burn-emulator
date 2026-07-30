# [[file:../../../../../../data/Nextcloud-SIG/dschmidt_working_projects/PC612_PS_RRM/PC612_emulator_v2.org::*Generate std data][Generate std data:1]]
import os
import numpy as np
import rasterio as rio

BASE = "/mnt/share/rem"
PROJ = "PC612_emulator_WC711_v1"
OUTPUT_DIR_NAME = "ignition_location_uncertainty"

md = 480
model = "circle"
data_source = "basemax"

num_iterations = 30 # number of reps in which ign location can vary (min reps req'd for CLT)

PS_SCENARIO = "s2"
PS_DATA_DIR = f"{BASE}/{PROJ}/PS_scenarios"
PS_OUTPUT_DIR = f"{PS_DATA_DIR}/{PS_SCENARIO}/{OUTPUT_DIR_NAME}/CNN_inference"
PS_proj_ids = range(1, 11) # s2 (could read from gpkg)

template_raster = rio.open(f"{PS_OUTPUT_DIR}/iteration_0/1/1/{model}_{data_source}_{md}_baseline.tif")
template_profile = template_raster.profile

for proj_id in PS_proj_ids:
    rd_cbp_arrays = []

    # read rasters needed for CBP
    for iteration in range(num_iterations):
        print(f"{proj_id=}, {iteration=}")

        baseline_fire_type_arrays = []
        legalmax_fire_type_arrays = []

        ignitions = sorted([d for d in os.listdir(f"{PS_OUTPUT_DIR}/iteration_{iteration}/{proj_id}") if os.path.isdir(f"{PS_OUTPUT_DIR}/iteration_{iteration}/{proj_id}/{d}")], key=int)

        for ignition in ignitions:
            if int(ignition) % 50 == 0:
                print(f"    {ignition=}")

            baseline_a = rio.open(f"{PS_OUTPUT_DIR}/iteration_{iteration}/{proj_id}/{ignition}/{model}_{data_source}_{md}_baseline.tif").read(1)
            legalmax_a = rio.open(f"{PS_OUTPUT_DIR}/iteration_{iteration}/{proj_id}/{ignition}/{model}_{data_source}_{md}_legalmax.tif").read(1)

            # hardcoding per MT (see 6/30 DM)
            baseline_a = np.where(baseline_a > 0.5, 1, 0)
            legalmax_a = np.where(legalmax_a > 0.5, 1, 0)

            baseline_fire_type_arrays.append(baseline_a)
            legalmax_fire_type_arrays.append(legalmax_a)

        # compute CBP
        baseline_cbp_a = np.sum(baseline_fire_type_arrays, axis=0) / len(ignitions)
        legalmax_cbp_a = np.sum(legalmax_fire_type_arrays, axis=0) / len(ignitions)

        with rio.open(f"{PS_OUTPUT_DIR}/iteration_{iteration}/{proj_id}/baseline_cbp.tif", "w", **template_profile) as dst:
            dst.write(baseline_cbp_a, 1)
        with rio.open(f"{PS_OUTPUT_DIR}/iteration_{iteration}/{proj_id}/legalmax_cbp.tif", "w", **template_profile) as dst:
            dst.write(legalmax_cbp_a, 1)
        print("    Finished saving CBP rasters")
        print("    Computing rel diff CBP")

        # compute relative difference CBP
        # mask for RD CBP calc and then set to max obs. value
        masked_baseline_cbp_a = np.ma.masked_where(baseline_cbp_a == 0, baseline_cbp_a)
        rd_cbp_a = (legalmax_cbp_a - masked_baseline_cbp_a) / masked_baseline_cbp_a
        rd_cbp_a[(baseline_cbp_a == 0) & (legalmax_cbp_a > 0)] = np.max(rd_cbp_a)
        #rd_cbp_a = np.ma.masked_where(rd_cbp_a == 0, rd_cbp_a) # only needed when symbolizing
        rd_cbp_arrays.append(rd_cbp_a)

        with rio.open(f"{PS_OUTPUT_DIR}/iteration_{iteration}/pa{proj_id}_rd_cbp.tif", "w", **template_profile) as dst:
            dst.write(rd_cbp_a.astype(rio.float32), 1)

        print("    Finished saving rel diff CBP raster")

    # compute mean and SD of relative difference CBP
    rd_cbp_std_a = np.std(rd_cbp_arrays, axis=0)
    rd_cbp_mean_a = np.average(rd_cbp_arrays, axis=0)
    print(f"{np.average(rd_cbp_std_a)=}, {np.average(rd_cbp_mean_a)=}") # average over the rep-averaged rasters

    with rio.open(f"{PS_OUTPUT_DIR}/pa{proj_id}_rd_cbp_std.tif", "w", **template_profile) as dst:
        dst.write(rd_cbp_std_a.astype(rio.float32), 1)

    with rio.open(f"{PS_OUTPUT_DIR}/pa{proj_id}_rd_cbp_mean.tif", "w", **template_profile) as dst:
        dst.write(rd_cbp_mean_a.astype(rio.float32), 1)
# Generate std data:1 ends here
