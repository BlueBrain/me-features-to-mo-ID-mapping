# This file is part of me-features-to-mol-ID-mapping.
#
#
# Copyright © 2021-2022 Blue Brain Project/EPFL
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the APACHE-2 License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

import getpass
from kgforge.core import KnowledgeGraphForge
import numpy as np

TOKEN = getpass.getpass(prompt="Enter NEXUS token")
# TOKEN = ""

# Target the sscx dissemination project in Nexus
ORG = "public"
PROJECT = "sscx"
nexus_endpoint = "https://bbp.epfl.ch/nexus/v1"

forge = KnowledgeGraphForge("https://raw.githubusercontent.com/BlueBrain/nexus-forge/master/examples/notebooks/use-cases/prod-forge-nexus.yml",
                            endpoint=nexus_endpoint,
                            bucket=f"{ORG}/{PROJECT}",
                            token=TOKEN
                            )
# Look for morphologies
print("seaching morphologies...")
path = forge.paths("Dataset") # to have autocompletion on the properties
mdata = forge.search(
                    path.type.id == "ReconstructedCell",
                    path.annotation.hasBody.type == "MType",
                    path.distribution.encodingFormat == "application/swc",
                    limit=2000
                    )

print(str(len(mdata)) + " dataset of type ReconstructedCell found.")
BBP_pyr_labels = ['L2_TPC:A', 'L2_TPC:B', 'L2_IPC',
                  'L3_TPC:A', 'L3_TPC:C',
                  'L4_TPC', 'L4_UPC', 'L4_SSC',
                  'L5_TPC:A', 'L5_TPC:B', 'L5_UPC', 'L5_TPC:C',
                  'L6_TPC:A', 'L6_TPC:C', 'L6_BPC', 'L6_HPC',  'L6_UPC', 'L6_IPC'
                  ]
mdata_df = forge.as_dataframe(mdata)
msk_m = np.asarray([mtype not in BBP_pyr_labels for mtype in mdata_df["annotation.hasBody.label"]])
mdata_df = mdata_df[msk_m]
print(str(len(mdata_df)) + " of inhibitory dataset of type ReconstructedCell found.")

for i, m in enumerate(mdata):
    if m.annotation.hasBody.label not in BBP_pyr_labels:
        forge.download(m, "distribution.contentUrl", "./BBP/BBP_morphologies/")

mdata_df.to_csv("./BBP/metadata_morphologies.csv")

# Look for electrophysiological recordings
print("seaching electrophysiological recordings...")

edata = forge.search({
    "type": "Trace",
    "distribution": {
        "encodingFormat": "application/nwb"
    },
    "contribution": {
        "hadRole": {
            "label": "neuron electrophysiology recording role"
        }
    },
    "note": "All traces"
},
limit = 500)

print(f"{str(len(edata))} data of type Trace found.")
edata_df = forge.as_dataframe(edata)
msk_e = np.asarray([etype not in ["cADpyr"] for etype in edata_df["annotation.hasBody.label"]])
edata_df = edata_df[msk_e]
print(str(len(edata_df)) + " of inhibitory data of type Trace found.")

# forge.download(edata, "distribution.contentUrl", "./BBP/BBP_ephys_traces/")
for i, e in enumerate(edata):
    if e.annotation.hasBody.label not in ["cADpyr"]:
        forge.download(e, "distribution.contentUrl", "./BBP/BBP_ephys_traces/")

edata_df.to_csv("./BBP/metadata_electrophysiology.csv")
