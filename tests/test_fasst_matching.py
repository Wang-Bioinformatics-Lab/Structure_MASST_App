"""FASST mode: MassBank records by peaks, and tying matched files to ReDU."""
import pandas as pd

from bin.massbank_spectra import accession_of, parse_record
from bin.workflow_stepwise import align_files_to_redu

RECORD = """ACCESSION: MSBNK-AAFC-AC000726
CH$NAME: Surfactin C
MS$FOCUSED_ION: PRECURSOR_M/Z 1036.6899
MS$FOCUSED_ION: PRECURSOR_TYPE [M+H]+
PK$NUM_PEAK: 3
PK$PEAK: m/z int. rel.int.
  72.0817 727290.9375 93
  84.0453 997138.1875 128
  86.0973 7688015.0 999
//
"""


def test_massbank_record_parsing():
    s = parse_record(RECORD)
    assert s["precursor_mz"] == 1036.6899
    assert s["peaks"] == [[72.0817, 727290.9375], [84.0453, 997138.1875], [86.0973, 7688015.0]]
    assert parse_record("PK$PEAK: m/z int. rel.int.\n  72.1 1 1\n//") is None  # no precursor


def test_massbank_accessions_only():
    assert accession_of("MSBNK-AAFC-AC000726") == "MSBNK-AAFC-AC000726"
    assert accession_of("mzspec:MASSBANK::accession:MSBNK-RIKEN_IMS-LQB00001") == "MSBNK-RIKEN_IMS-LQB00001"
    for bad in ("CCMSLIB00000531487", "MSBNK-../../etc-x", "MSBNK-AAFC-AC0007/26", ""):
        assert accession_of(bad) is None


REDU = pd.Series([
    "mzspec:MSV1:peak/mzXML/PLATE_3/S_57718.mzXML",                    # MassIVE original of a ccms_peak copy
    "mzspec:MSV2:updates/2025-01-30_x_1/peak/MSV2_mzML/sub/Run54.mzML",  # re-upload
    "mzspec:MSV3:pos/QC_01.mzML", "mzspec:MSV3:neg/QC_01.mzML",           # same name twice
    "mzspec:MSV4:raw/A.mzML",
    "mzspec:MSV6:peak/RP/S1.mzXML",                                     # one entry, matched twice (RP and HILIC)
    "mzspec:MSV7:peak/X.mzXML",                                         # matched as two copies
    "mzspec:MSV8:ccms_peak/So1/A2.mzML",                                # So1 matched exactly, So13 is another file
])


def test_files_tied_to_redu():
    files = pd.Series([
        "mzspec:MSV1:ccms_peak/mzXML/PLATE_3/S_57718.mzML",   # conversion layout
        "mzspec:MSV2:gnps2_converted/raw/Run54.mzML",          # unique name in the dataset
        "mzspec:MSV3:ccms_peak/neg/QC_01.mzML",                # path decides between pos and neg
        "mzspec:MSV3:gnps2_converted/QC_01.mzML",              # name alone is ambiguous: left alone
        "mzspec:MSV4:raw/A.mzML",                              # exact
        "mzspec:MSV5:ccms_peak/B.mzML",                        # dataset not in ReDU
        None,
        "mzspec:MSV6:ccms_peak/data/RP/S1.mzML",               # the RP file gets the RP entry ...
        "mzspec:MSV6:ccms_peak/data/HILIC/S1.mzML",            # ... the HILIC file nothing
        "mzspec:MSV7:ccms_peak/X.mzML",                        # copies of one file: both
        "mzspec:MSV7:gnps2_converted/X.mzML",
        "mzspec:MSV8:ccms_peak/So1/A2.mzML",                   # exact
        "mzspec:MSV8:ccms_peak/So13/A2.mzML",                  # same name, other folder, entry taken: left alone
    ])
    out = align_files_to_redu(files, REDU).tolist()
    assert out == [
        "mzspec:MSV1:peak/mzXML/PLATE_3/S_57718.mzXML",
        "mzspec:MSV2:updates/2025-01-30_x_1/peak/MSV2_mzML/sub/Run54.mzML",
        "mzspec:MSV3:neg/QC_01.mzML",
        "mzspec:MSV3:gnps2_converted/QC_01.mzML",
        "mzspec:MSV4:raw/A.mzML",
        "mzspec:MSV5:ccms_peak/B.mzML",
        None,
        "mzspec:MSV6:peak/RP/S1.mzXML",
        "mzspec:MSV6:ccms_peak/data/HILIC/S1.mzML",
        "mzspec:MSV7:peak/X.mzXML",
        "mzspec:MSV7:peak/X.mzXML",
        "mzspec:MSV8:ccms_peak/So1/A2.mzML",
        "mzspec:MSV8:ccms_peak/So13/A2.mzML",
    ]
