"""Build the consolidated HAB-relevant CSV from CMFRI Annual Reports.

Long format: one row per (report, region, sub_location, season/month, depth, variable).
Values captured as reported; unit column preserves source unit notation.

Sources:
- CMFRI Annual Report 2016-17 pp.54, 55, 86  (Mumbai Jan-Dec 2016; Vizag 2012-17 narrative)
- CMFRI Annual Report 2017-18 (AR 2018)  p.51  (Karnataka/Goa purse-seine grounds)
- CMFRI Annual Report 2020 p.162  (Vembanad Lake / Cochin backwaters)
- CMFRI Annual Report 2022 p.141  (Mangaluru Iddya bloom; Noctiluca effects note)
- CMFRI Annual Report 2023 pp.40-43, 122  (Off Cochin/Mandapam; Veraval sea cage)
- CMFRI Annual Report 2024 pp.86, 162  (Visakhapatnam Nov-Dec 2024; Kochi Trichodesmium bloom)
"""

import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data"
import csv, pathlib

OUT = pathlib.Path(str(_DATA_DIR / "cmfri_hab_env_dataset.csv"))

HEADER = [
    "source_report", "data_year", "region", "sub_location",
    "season_or_month", "depth_or_layer",
    "variable", "value_min", "value_max", "value_mean",
    "units", "dominant_taxa_or_notes",
]

rows = []

def add(rep, yr, reg, sub, seas, dep, var, vmin="", vmax="", vmean="", units="", notes=""):
    rows.append([rep, yr, reg, sub, seas, dep, var, vmin, vmax, vmean, units, notes])

# ------------------------------------------------------------------
# 2016-17 report p.54 : Mumbai creek + near-shore x pre/monsoon/post-monsoon (Jan-Dec 2016)
# ------------------------------------------------------------------
REP1 = "CMFRI AR 2016-17"; YR1 = 2016; REG1 = "Mumbai (Maharashtra)"
mumbai_p54 = {
    # (variable, units): {(sub, season): (min, max, mean)}
    ("air_temperature", "C"): {
        ("creek","pre_monsoon"):(25.0,37.0,33.4),
        ("creek","monsoon"):(26.0,35.0,30.0),
        ("creek","post_monsoon"):(29.0,37.0,32.8),
        ("near_shore","pre_monsoon"):(25.0,37.1,32.6),
        ("near_shore","monsoon"):(24.0,34.5,30.4),
        ("near_shore","post_monsoon"):(24.6,35.0,30.0),
    },
    ("sea_surface_temperature","C"): {
        ("creek","pre_monsoon"):(23.6,32.8,30.3),
        ("creek","monsoon"):(26.0,33.0,28.4),
        ("creek","post_monsoon"):(24.8,29.8,27.0),
        ("near_shore","pre_monsoon"):(23.3,32.8,29.5),
        ("near_shore","monsoon"):(25.4,32.3,28.8),
        ("near_shore","post_monsoon"):(24.0,35.0,30.0),
    },
    ("salinity","ppt"): {
        ("creek","pre_monsoon"):(3.2,34.2,21.6),
        ("creek","monsoon"):(0.5,33.5,8.8),
        ("creek","post_monsoon"):(12.9,33.8,24.8),
        ("near_shore","pre_monsoon"):(24.6,35.5,32.0),
        ("near_shore","monsoon"):(2.0,34.3,21.5),
        ("near_shore","post_monsoon"):(28.8,34.8,33.4),
    },
    ("dissolved_oxygen","mg/L"): {
        ("creek","pre_monsoon"):(0.2,6.0,2.3),
        ("creek","monsoon"):(0.9,3.8,2.4),
        ("creek","post_monsoon"):(0.4,4.2,1.7),
        ("near_shore","pre_monsoon"):(1.2,6.8,3.2),
        ("near_shore","monsoon"):(1.6,6.4,4.1),
        ("near_shore","post_monsoon"):(1.6,6.1,3.1),
    },
    ("pH","unitless"): {
        ("creek","pre_monsoon"):(6.4,8.13,""),
        ("creek","monsoon"):(6.7,8.4,""),
        ("creek","post_monsoon"):(6.5,8.3,""),
        ("near_shore","pre_monsoon"):(6.4,8.5,""),
        ("near_shore","monsoon"):(6.4,8.3,""),
        ("near_shore","post_monsoon"):(7.2,8.4,""),
    },
    ("BOD","mg/L"): {
        ("creek","pre_monsoon"):(4.0,12.0,7.5),
        ("creek","monsoon"):(2.0,11.0,7.9),
        ("creek","post_monsoon"):(5.0,12.0,8.2),
        ("near_shore","pre_monsoon"):(2.0,6.0,3.3),
        ("near_shore","monsoon"):(2.0,8.0,4.5),
        ("near_shore","post_monsoon"):(2.0,9.0,4.5),
    },
    ("chlorophyll_a","mg/m3"): {
        ("creek","pre_monsoon"):(3.5,9.9,7.8),
        ("creek","monsoon"):(4.2,26.9,14.8),
        ("creek","post_monsoon"):(2.5,16.5,10.1),
        ("near_shore","pre_monsoon"):(0.16,14.7,5.8),
        ("near_shore","monsoon"):(0.8,10.7,7.2),
        ("near_shore","post_monsoon"):(5.0,15.4,9.4),
    },
    ("phosphate","mg/L"): {
        ("creek","pre_monsoon"):(0.7,7.2,2.0),
        ("creek","monsoon"):(0.2,6.4,1.4),
        ("creek","post_monsoon"):(0.05,8.7,4.1),
        ("near_shore","pre_monsoon"):(0.4,4.8,1.8),
        ("near_shore","monsoon"):(0.3,5.4,1.7),
        ("near_shore","post_monsoon"):(0.04,14.6,2.3),
    },
    ("nitrate","mg/L"): {
        ("creek","pre_monsoon"):(3.2,7.8,4.8),
        ("creek","monsoon"):(1.4,7.2,4.5),
        ("creek","post_monsoon"):(1.8,14.5,5.5),
        ("near_shore","pre_monsoon"):(2.7,5.6,3.9),
        ("near_shore","monsoon"):(2.1,9.9,5.1),
        ("near_shore","post_monsoon"):(1.6,14.6,5.9),
    },
    ("nitrite","mg/L"): {
        ("creek","pre_monsoon"):(0.06,0.6,0.1),
        ("creek","monsoon"):(0.08,1.6,0.38),
        ("creek","post_monsoon"):(0.07,1.5,0.4),
        ("near_shore","pre_monsoon"):(0.07,1.0,0.2),
        ("near_shore","monsoon"):(0.1,1.6,0.4),
        ("near_shore","post_monsoon"):(0.07,2.0,0.7),
    },
    ("silicate","mg/L"): {
        ("creek","pre_monsoon"):(2.4,8.1,4.5),
        ("creek","monsoon"):(0.1,7.1,2.0),
        ("creek","post_monsoon"):(0.49,10.7,6.5),
        ("near_shore","pre_monsoon"):(1.1,4.0,2.8),
        ("near_shore","monsoon"):(0.2,5.1,2.0),
        ("near_shore","post_monsoon"):(1.0,8.1,2.9),
    },
    ("ammonia","mg/L"): {
        ("creek","pre_monsoon"):(0.3,1.8,1.0),
        ("creek","monsoon"):(0.06,1.2,0.47),
        ("creek","post_monsoon"):(0.06,3.6,1.5),
        ("near_shore","pre_monsoon"):(0.06,1.0,0.4),
        ("near_shore","monsoon"):(0.09,1.1,0.4),
        ("near_shore","post_monsoon"):(0.03,1.0,0.4),
    },
    ("turbidity","NTU"): {
        ("creek","pre_monsoon"):(0.9,91.7,21.9),
        ("creek","monsoon"):(37.6,172.0,78.8),
        ("creek","post_monsoon"):(5.6,64.8,27.9),
        ("near_shore","pre_monsoon"):(1.6,178,46.8),
        ("near_shore","monsoon"):(21.6,355.0,91.2),
        ("near_shore","post_monsoon"):(11.8,227.0,59.1),
    },
    ("TSS","mg/L"): {
        ("creek","pre_monsoon"):(0.08,0.7,0.2),
        ("creek","monsoon"):(0.3,0.7,0.4),
        ("creek","post_monsoon"):(0.12,0.5,0.3),
        ("near_shore","pre_monsoon"):(0.12,0.7,0.3),
        ("near_shore","monsoon"):(0.04,0.6,0.2),
        ("near_shore","post_monsoon"):(0.06,0.66,0.2),
    },
    ("TDS","ppt"): {
        ("creek","pre_monsoon"):(3.3,28.7,11.2),
        ("creek","monsoon"):(1.3,36.3,8.4),
        ("creek","post_monsoon"):(1.6,4.3,2.5),
        ("near_shore","pre_monsoon"):(5.2,34.9,18.0),
        ("near_shore","monsoon"):(1.4,43.0,16.4),
        ("near_shore","post_monsoon"):(1.58,176.0,34.7),
    },
}
for (var, units), inner in mumbai_p54.items():
    for (sub, seas), (vmin, vmax, vmean) in inner.items():
        add(REP1, YR1, REG1, sub, seas, "surface", var, vmin, vmax, vmean, units,
            "6 stations: 3 creeks + 3 near-shore; Jan-Dec 2016; report p.54")

# 2016-17 p.55 : Mumbai by depth (10m/20m/30m), Jan-Dec 2016
mumbai_p55 = {
    ("air_temperature","C"): {"10m":(27.0,33.0,""),"20m":(27.0,36.0,""),"30m":(25.0,34.0,"")},
    ("sea_surface_temperature","C"): {"10m":(25.3,29.7,""),"20m":(26.5,29.4,""),"30m":(25.0,29.6,"")},
    ("salinity","ppt"): {"10m":(32.04,35.3,""),"20m":(32.6,35.4,""),"30m":(31.6,35.2,"")},
    ("dissolved_oxygen","mg/L"): {"10m":(3.91,7.9,""),"20m":(4.34,6.40,""),"30m":(2.07,6.02,"")},
    ("pH","unitless"): {"10m":(8.2,8.77,""),"20m":(8.12,8.71,""),"30m":(8.23,8.70,"")},
    ("BOD","mg/L"): {"10m":(0,2,""),"20m":(0.0,4.0,""),"30m":(0,2,"")},
    ("chlorophyll_a","mg/m3"): {"10m":(5.73,6.81,""),"20m":(4.50,5.09,""),"30m":(4.98,6.89,"")},
    ("phosphate","mg/L"): {"10m":(1.3,1.8,""),"20m":(0.90,1.10,""),"30m":(0.9,1.2,"")},
    ("nitrate","mg/L"): {"10m":(2.71,4.9,""),"20m":(6.42,8.3,""),"30m":(5.31,6.1,"")},
    ("nitrite","mg/L"): {"10m":(0.29,0.45,""),"20m":(0.10,0.18,""),"30m":(0.17,0.29,"")},
    ("silicate","mg/L"): {"10m":(3.01,3.80,""),"20m":(2.35,3.09,""),"30m":(1.29,1.86,"")},
    ("ammonia","mg/L"): {"10m":(0.39,0.59,""),"20m":(0.54,1.54,""),"30m":(0.22,0.36,"")},
    ("turbidity","NTU"): {"10m":(8.58,33.2,""),"20m":(4.14,14.75,""),"30m":(1.84,3.85,"")},
    ("TSS","mg/L"): {"10m":(0.154,0.414,""),"20m":(0.23,0.36,""),"30m":(0.21,0.26,"")},
    ("TDS","ppt"): {"10m":(6.25,28.9,""),"20m":(5.98,33.93,""),"30m":(6.12,35.49,"")},
}
for (var, units), inner in mumbai_p55.items():
    for depth, (vmin, vmax, vmean) in inner.items():
        add(REP1, YR1, REG1, "offshore_depth_transect", "annual", depth, var,
            vmin, vmax, vmean, units,
            "Depth-stratified Mumbai Jan-Dec 2016; report p.55")

# Note dominant taxa
add(REP1, YR1, REG1, "Mahim_Creek", "annual", "surface",
    "phytoplankton_dominant_taxa", "", "", "", "species",
    "Thalassiosira subtilis, Coscinodiscus granii, Navicula distans, Skeletonema costatum; highest species count in June 2016; p.55")

# 2016-17 p.86 : Vizag (Andhra Pradesh) 5-year narrative (Apr 2012 - Mar 2017)
REG_VIZ = "Visakhapatnam (Andhra Pradesh)"
add(REP1, 2012, REG_VIZ, "5yr_summary_Apr2012_Mar2017", "annual_min_Dec", "surface",
    "sea_surface_temperature", 25.7, "", "", "C",
    "Min SST occurred in December; from 5-yr compiled dataset; p.86")
add(REP1, 2012, REG_VIZ, "5yr_summary_Apr2012_Mar2017", "annual_max_May", "surface",
    "sea_surface_temperature", "", 30.43, "", "C",
    "Max SST occurred in May; p.86")
add(REP1, 2012, REG_VIZ, "5yr_summary_Apr2012_Mar2017", "annual_max_Apr", "surface",
    "salinity", "", 34.14, "", "ppt", "Max salinity April; p.86")
add(REP1, 2012, REG_VIZ, "5yr_summary_Apr2012_Mar2017", "annual_min_Oct", "surface",
    "salinity", 27.74, "", "", "ppt", "Min salinity October; p.86")
add(REP1, 2012, REG_VIZ, "5yr_summary_Apr2012_Mar2017", "Mar_Apr", "surface",
    "pH", 8.14, 8.19, "", "unitless", "Peak pH summer months; p.86")
add(REP1, 2012, REG_VIZ, "5yr_summary_Apr2012_Mar2017", "Sep_Nov", "surface",
    "pH", 7.96, "", "", "unitless", "Lowest pH Sep-Nov; p.86")

# ------------------------------------------------------------------
# 2017-18 (AR 2018) p.51 : Karnataka/Goa purse-seine grounds (single-value)
# ------------------------------------------------------------------
REP2 = "CMFRI AR 2017-18"; YR2 = 2017; REG2 = "Karnataka & Goa (purse-seine grounds)"
row_notes = ("Purse-seine day / full-moon nights / with artificial lights fishing grounds. "
             "Report writes 'µg at l-1' — historic notation for µg-atoms/L (i.e. µmol/L); "
             "reproduced verbatim in units column. Report p.51")
add(REP2, YR2, REG2, "", "annual", "surface", "pH", "", "", 8.2, "unitless", row_notes)
add(REP2, YR2, REG2, "", "annual", "surface", "salinity", "", "", 32.4, "ppt", row_notes)
add(REP2, YR2, REG2, "", "annual", "surface", "phosphate", "", "", 2.6, "ug_at/L", row_notes)
add(REP2, YR2, REG2, "", "annual", "surface", "silicate", "", "", 1.819, "ug_at/L", row_notes)
add(REP2, YR2, REG2, "", "annual", "surface", "nitrite", "", "", 0.187, "ug_at/L", row_notes)
add(REP2, YR2, REG2, "", "annual", "surface", "nitrate", "", "", 0.771, "ug_at/L", row_notes)
add(REP2, YR2, REG2, "", "annual", "surface", "ammonia", "", "", 0.166, "ug_at/L", row_notes)
add(REP2, YR2, REG2, "", "annual", "surface", "TSS", "", "", 43.11, "mg/L", row_notes)
add(REP2, YR2, REG2, "", "annual", "surface", "chlorophyll_a", "", "", 1.14, "mg/m3", row_notes)
add(REP2, YR2, REG2, "", "annual", "surface", "chlorophyll_b", "", "", 0.143, "mg/m3", row_notes)
add(REP2, YR2, REG2, "", "annual", "surface", "chlorophyll_c", "", "", 0.82, "mg/m3", row_notes)

# ------------------------------------------------------------------
# 2020 report p.162 : Vembanad Lake / Cochin backwaters
# ------------------------------------------------------------------
REP3 = "CMFRI AR 2020"; YR3 = 2020; REG3 = "Vembanad Lake / Cochin backwaters (Kerala)"
add(REP3, YR3, REG3, "11_stations_Edayar_Aroor", "monsoon", "surface",
    "chlorophyll_a", 0.88, 3.61, "", "mg/m3", "p.162")
add(REP3, YR3, REG3, "11_stations_Edayar_Aroor", "post_monsoon", "surface",
    "chlorophyll_a", 5.2, 18.47, "", "mg/m3", "p.162")
add(REP3, YR3, REG3, "11_stations_Edayar_Aroor", "post_monsoon", "surface",
    "ammonia", 0.21, 0.35, "", "mg/L", "TAN (total ammoniacal N); p.162")
add(REP3, YR3, REG3, "Thopumpady", "monsoon", "surface",
    "phytoplankton_abundance", "", "", 2100000, "cells/L",
    "Pleurosigma elongatum dominance driving high turbidity; p.162")

# ------------------------------------------------------------------
# 2022 report p.141 : Mangaluru Iddya diatom bloom + Noctiluca effects
# ------------------------------------------------------------------
REP4 = "CMFRI AR 2022"; YR4 = 2022
REG4a = "Mangaluru / Iddya (Karnataka)"
add(REP4, YR4, REG4a, "Iddya_Surathkal", "2022-05-14", "surface",
    "phytoplankton_abundance", "", "", 1.20e8, "cells/L",
    "Diatoma vulgaris bloom (1.20x10^5 cells/mL converted to cells/L); p.141")
REG4b = "coastal (multiple stations)"
add(REP4, YR4, REG4b, "Noctiluca_scintillans_bloom_areas", "bloom_period", "surface",
    "bloom_effect_note", "", "", "", "qualitative",
    "Noctiluca scintillans bloom -> DO reduced; chlorophyll, ammonia-N, nitrate-N, nitrite-N, phosphate-P, silicate-Si all rose with algal density; p.141")

# ------------------------------------------------------------------
# 2023 report pp.40-43 : off Cochin & Mandapam water quality (2023)
# ------------------------------------------------------------------
REP5 = "CMFRI AR 2023"; YR5 = 2023
REG5a = "off Cochin (Kerala)"
# Annual means
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "annual", "surface",
    "sea_surface_temperature", "", "", 29.9, "C", "±0.99; p.40")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "annual", "surface",
    "salinity", "", "", 31.85, "ppt", "±6.4; p.40")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "annual", "surface",
    "pH", "", "", 8.30, "unitless", "±0.19; p.40")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "annual", "surface",
    "dissolved_oxygen", "", "", 6.78, "mg/L", "±0.93; p.40")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "annual", "surface",
    "turbidity", "", "", 4.43, "NTU", "±4.09; p.40")
# Ranges / month-wise extremes
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "annual_range", "surface",
    "sea_surface_temperature", 28.0, 31.3, "", "C", "p.40")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "May", "surface",
    "sea_surface_temperature", "", 30.93, "", "C", "±0.12 monthly mean max; p.40")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "January", "surface",
    "sea_surface_temperature", 28.3, "", "", "C", "±0.3 monthly mean min; p.40")
add(REP5, YR5, REG5a, "Station_1_5m", "annual", "surface",
    "salinity", 7.0, "", "", "ppt", "min salinity at riverine-influenced St1; p.40")
add(REP5, YR5, REG5a, "Stations_2_3", "annual", "surface",
    "salinity", 27.0, 36.0, "", "ppt", "range at St2 and St3; p.40")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "April", "surface",
    "salinity", "", 35.33, "", "ppt", "±0.47 monthly mean max; p.41")
add(REP5, YR5, REG5a, "Station_2_10m", "June", "surface",
    "pH", 7.7, "", "", "unitless", "p.41")
add(REP5, YR5, REG5a, "Station_1_5m", "June", "surface",
    "pH", "", 8.6, "", "unitless", "p.41")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "October", "surface",
    "pH", "", "", 8.5, "unitless", "±0.05 monthly mean max; p.41")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "June", "surface",
    "pH", "", "", 8.15, "unitless", "±0.63 monthly mean min; p.41")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "April", "surface",
    "dissolved_oxygen", "", 7.57, "", "mg/L", "±0.78 monthly mean max; p.41")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "June", "surface",
    "dissolved_oxygen", 5.84, "", "", "mg/L", "±0.11 monthly mean min; p.41")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "annual_range", "surface",
    "dissolved_oxygen", 4.56, 8.48, "", "mg/L", "p.41")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "June", "surface",
    "turbidity", "", 9.35, "", "NTU", "±7.84 monthly max; p.41")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "October", "surface",
    "turbidity", 2.70, "", "", "NTU", "±1.31 monthly min; p.41")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "annual_range", "surface",
    "chlorophyll_a", 0.63, 15.81, 5.28, "ug/L",
    "avg ±3.93; min at St1 June, max at St3 March; p.42")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "annual", "surface",
    "phytoplankton_abundance", "", "", 240544, "cells/L",
    "April 2023 peak Cochin waters (Diatoms 454007; Cyano 183990; Dino 26894); p.41")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "January", "surface",
    "phytoplankton_abundance", "", "", 227633, "cells/L", "p.41")
add(REP5, YR5, REG5a, "3_stations_5m_10m_20m", "December", "surface",
    "phytoplankton_abundance", "", "", 114858, "cells/L", "p.41")
# Trichodesmium bloom Apr 2023 at St3
add(REP5, YR5, REG5a, "Station_3_20m_bloom", "April_2023", "surface",
    "phytoplankton_abundance", "", "", 114200, "cells/L",
    "Trichodesmium sp. bloom; brownish-red water; p.42")
add(REP5, YR5, REG5a, "Station_3_20m_bloom", "April_2023", "surface",
    "salinity", 35.0, 36.0, "", "ppt", "bloom period; p.42")
add(REP5, YR5, REG5a, "Station_3_20m_bloom", "April_2023", "surface",
    "sea_surface_temperature", 29.8, 31.2, "", "C", "bloom period; p.42")
add(REP5, YR5, REG5a, "Station_3_20m_bloom", "April_2023", "surface",
    "nitrate", "", "", 2.11, "umol/L", "bloom period; p.42")
add(REP5, YR5, REG5a, "Station_3_20m_bloom", "April_2023", "surface",
    "phosphate", "", "", 3.00, "umol/L", "bloom period; p.42")
add(REP5, YR5, REG5a, "Station_3_20m_bloom", "April_2023", "surface",
    "chlorophyll_a", "", "", 4.78, "ug/L", "bloom station relatively high chl-a; p.42")

REG5b = "Mandapam coast (Tamil Nadu)"
add(REP5, YR5, REG5b, "3_stations", "annual", "surface",
    "sea_surface_temperature", "", "", 27.40, "C", "±1.81; p.41")
add(REP5, YR5, REG5b, "3_stations", "annual", "surface",
    "salinity", "", "", 30.76, "ppt", "±2.81; p.41")
add(REP5, YR5, REG5b, "3_stations", "annual", "surface",
    "pH", "", "", 8.21, "unitless", "±0.13; p.41")
add(REP5, YR5, REG5b, "3_stations", "annual", "surface",
    "dissolved_oxygen", "", "", 6.42, "mg/L", "±0.89; p.41")
add(REP5, YR5, REG5b, "3_stations", "annual", "surface",
    "turbidity", "", "", 2.11, "NTU", "±1.58; p.41")
add(REP5, YR5, REG5b, "3_stations", "annual_range", "surface",
    "sea_surface_temperature", 25.0, 30.3, "", "C", "p.41")
add(REP5, YR5, REG5b, "3_stations", "January", "surface",
    "sea_surface_temperature", "", 30.20, "", "C", "±0.08 monthly mean max; p.41")
add(REP5, YR5, REG5b, "3_stations", "September", "surface",
    "sea_surface_temperature", 25.66, "", "", "C", "±0.47 monthly mean min; p.41")
add(REP5, YR5, REG5b, "3_stations", "annual_range", "surface",
    "salinity", 26.5, 34.0, "", "ppt", "in Jan-Feb; p.41")
add(REP5, YR5, REG5b, "3_stations", "annual_range", "surface",
    "dissolved_oxygen", 4.9, 7.5, "", "mg/L", "p.41")
add(REP5, YR5, REG5b, "3_stations", "February", "surface",
    "dissolved_oxygen", "", 7.33, "", "mg/L", "±0.13 monthly mean max; p.41")
add(REP5, YR5, REG5b, "3_stations", "September", "surface",
    "dissolved_oxygen", 5.16, "", "", "mg/L", "±0.20 monthly mean min; p.41")
add(REP5, YR5, REG5b, "3_stations", "annual_range", "surface",
    "turbidity", 2.96, 3.36, "", "NTU", "Feb 2.96±0.74 to Sep 3.36±0.41; p.41")
add(REP5, YR5, REG5b, "3_stations", "annual", "surface",
    "chlorophyll_a", 0.98, 1.84, 1.46, "ug/L",
    "avg ±0.31; min Jan St3, max Feb St1; p.42")
add(REP5, YR5, REG5b, "3_stations", "January", "surface",
    "phytoplankton_abundance", "", "", 14322, "cells/L",
    "Diatoms 14734; Cyano 9682 dominant Mandapam; p.41")
add(REP5, YR5, REG5b, "3_stations", "February", "surface",
    "phytoplankton_abundance", "", "", 7505, "cells/L", "p.41")
add(REP5, YR5, REG5b, "3_stations", "September", "surface",
    "phytoplankton_abundance", "", "", 2723, "cells/L", "p.41")

# 2023 p.122 : Veraval sea cage farm
REG5c = "off Veraval (Gujarat)"
add(REP5, YR5, REG5c, "sea_cage_farm_600m_from_shore", "Oct_to_Mar", "8.72m_depth",
    "TSS", 344.0, 482.08, "", "mg/L", "at culture site; p.122")
add(REP5, YR5, REG5c, "sea_cage_farm", "seasonal", "8.72m_depth",
    "dissolved_oxygen", 4.35, 5.93, "", "mg/L", "culture site range; p.122")
add(REP5, YR5, REG5c, "reference_site_1000m_from_shore", "seasonal", "8.8m_depth",
    "dissolved_oxygen", 4.78, 6.24, "", "mg/L", "reference site range; p.122")
add(REP5, YR5, REG5c, "sea_cage_farm", "seasonal", "8.72m_depth",
    "gross_primary_productivity", 0.08, 0.23, "", "mg_C/L/hr", "culture site; p.122")
add(REP5, YR5, REG5c, "sea_cage_farm", "seasonal", "8.72m_depth",
    "ammonia", 0.09, 0.47, "", "ug_NH4-N/L", "culture site; p.122")
add(REP5, YR5, REG5c, "reference_site_1000m_from_shore", "seasonal", "8.8m_depth",
    "ammonia", 0.20, 0.98, "", "ug_NH4-N/L", "reference site; p.122")

# ------------------------------------------------------------------
# 2024 report p.86 : Visakhapatnam Nov & Dec 2024 (3 sites)
# ------------------------------------------------------------------
REP6 = "CMFRI AR 2024"; YR6 = 2024; REG6 = "Visakhapatnam (Andhra Pradesh)"
# Chlorophyll extremes
add(REP6, YR6, REG6, "Kailashgiri", "December", "surface",
    "chlorophyll_a", 0.4531, "", "", "mg/m3", "monthly minimum surface chl-a in Dec; p.86")
add(REP6, YR6, REG6, "R_B_Beach_cage_site", "November", "surface",
    "chlorophyll_a", "", 2.2584, "", "mg/m3", "monthly maximum surface chl-a in Nov; p.86")
add(REP6, YR6, REG6, "Yarada", "November", "bottom",
    "chlorophyll_a", "", 1.932, "", "mg/m3", "monthly max bottom chl-a in Nov; p.86")
add(REP6, YR6, REG6, "Yarada", "December", "bottom",
    "chlorophyll_a", 0.3678, "", "", "mg/m3", "monthly min bottom chl-a in Dec; p.86")
# Salinity
add(REP6, YR6, REG6, "3_sites_composite", "November", "surface",
    "salinity", "", "", 22, "ppt", "surface & bottom both 22 ppt; p.86")
add(REP6, YR6, REG6, "3_sites_composite", "November", "bottom",
    "salinity", "", "", 22, "ppt", "surface & bottom both 22 ppt; p.86")
add(REP6, YR6, REG6, "3_sites_composite", "December", "surface",
    "salinity", 28, 29, "", "ppt", "p.86")
add(REP6, YR6, REG6, "3_sites_composite", "December", "bottom",
    "salinity", 28, 29, "", "ppt", "p.86")
# pH
add(REP6, YR6, REG6, "3_sites_composite", "Nov_Dec", "surface",
    "pH", 8.0, 8.2, 8.1, "unitless", "both months; p.86")
add(REP6, YR6, REG6, "3_sites_composite", "Nov_Dec", "bottom",
    "pH", 8.0, 8.2, 8.1, "unitless", "both months; p.86")
# DO
add(REP6, YR6, REG6, "3_sites_composite", "November", "surface",
    "dissolved_oxygen", "", "", 5.58, "mg/L", "average surface DO; p.86")
add(REP6, YR6, REG6, "3_sites_composite", "December", "surface",
    "dissolved_oxygen", "", "", 5.54, "mg/L", "average surface DO; p.86")
add(REP6, YR6, REG6, "3_sites_composite", "Nov_Dec", "bottom",
    "dissolved_oxygen", "", 5, "", "mg/L", "sub-surface DO < 5 mg/L both months; p.86")
# Ortho-phosphate
add(REP6, YR6, REG6, "Kailashgiri", "December", "surface",
    "ortho_phosphate", "", 0.04, "", "mg/L", "highest surface P Dec; p.86")
add(REP6, YR6, REG6, "Yarada", "December", "surface",
    "ortho_phosphate", 0.002, "", "", "mg/L", "lowest surface P Dec; p.86")

# 2024 p.162 : Trichodesmium bloom off Kochi, March 2024
REG6b = "off Kochi (Kerala)"
add(REP6, YR6, REG6b, "algal_bloom_patch", "2024-03", "surface",
    "phytoplankton_abundance", "", "", 1522000, "cells/L",
    "Trichodesmium bloom (15.22 lakh cells/L); triggered by elevated SST, nutrient enrichment, low phosphate; p.162")
add(REP6, YR6, REG6b, "algal_bloom_patch", "2024-03", "surface",
    "CDOM_absorbance_440nm", "", "", 0.277, "1/m",
    "Elevated at bloom vs non-bloom (0.16-0.20/m); p.162")
add(REP6, YR6, REG6b, "algal_bloom_patch", "2024-03", "surface",
    "NDCI_Sentinel2", "", "", 0.3, "unitless",
    "NDCI 0.3 = moderately high chlorophyll-a; from Sentinel-2 08 Mar 2024; p.162")
add(REP6, YR6, REG6b, "algal_bloom_patch", "2024-03", "surface",
    "bloom_flag", "", "", "yes", "categorical",
    "Trichodesmium sp. bloom event; p.162")

# ------------------------------------------------------------------
# Optional: Upwelling narrative rows so upwelling isn't lost
# ------------------------------------------------------------------
add(REP2, 2017, "Kochi coast (Kerala)", "SEAS_upwelling_index", "annual", "surface",
    "peak_upwelling_month", "", "", "August_2017", "categorical",
    "Peak upwelling August 2017 (from prior project context / SEAS is upwelling-dominated); ref CMFRI AR 2017-18")
add("CMFRI AR 2019", 2018, "SW coast India / SEAS", "EMT_upwelling_index", "summer_monsoon", "surface",
    "upwelling_correlation_with_chl_a", "", "", 0.5, "Pearson_r",
    "Chl-a vs Ekman Mass Transport r=0.5 (higher than precipitation r=0.39); 1998-2016 satellite; report p.36")

# Sort by year for readability
rows.sort(key=lambda r: (str(r[1]), r[0], r[2], r[3], r[6]))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(HEADER)
    w.writerows(rows)
print(f"wrote {OUT} with {len(rows)} rows")
