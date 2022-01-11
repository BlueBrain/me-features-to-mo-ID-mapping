import pandas as pd
import glob
import os
import shutil
from allensdk.core.cell_types_cache import CellTypesCache

AIBS_meta = pd.read_excel(io='41593_2019_417_MOESM5_ESM.xlsx', sheet_name='Data',
                          index_col='specimen_id')
mask_Inh = ['Inh' in str(x) for x in AIBS_meta['me-type'].values]

# Instantiate the CellTypesCache instance.  The manifest_file argument
# tells it where to store the manifest, which is a JSON file that tracks
# file paths.  If you supply a relative path (like this), it will go
# into your current working directory
ctc = CellTypesCache(manifest_file='Gouwens_2019/manifest.json')

for i, cell_id in enumerate(AIBS_meta.index[mask_Inh]):
    
    if (str(AIBS_meta['e-type'][cell_id]) != 'nan') & (str(AIBS_meta['m-type'][cell_id]) != 'nan'):
        try:
            # this saves the NWB file to 'cell_types/specimen_464212183/ephys.nwb'
            data_set = ctc.get_ephys_data(cell_id)
            # download and open an SWC file
            morphology = ctc.get_reconstruction(cell_id)
        except:
            print("no data for ", str(cell_id))

    # if (i > 9):
    #     break

print("rearanging directory...")
path = "./Gouwens_2019/*"
for i, file in enumerate(glob.glob(path)):

    if ("manifest.json" not in file)&("ephys_traces" not in file)&("morphologies" not in file):

        try:
            efile = file + "/ephys.nwb"
            new_ename = "_".join(efile.split("/")[-2:])

            mfile = file + "/reconstruction.swc"
            new_mname = "_".join(mfile.split("/")[-2:])

            if not os.path.isdir("./Gouwens_2019/ephys_traces/"):
                os.mkdir("./Gouwens_2019/ephys_traces/")

            if not os.path.isdir("./Gouwens_2019/morphologies/"):
                os.mkdir("./Gouwens_2019/morphologies/")

            shutil.move(efile, "./Gouwens_2019/ephys_traces/" + new_ename)
            shutil.move(mfile, "./Gouwens_2019/morphologies/" + new_mname)
        except:
            print("no data for ", file)

        os.rmdir(file)
