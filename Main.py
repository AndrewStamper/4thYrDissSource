"""
This is the main API which calls all subsequent actions
"""

from Experiments.AnalyseScan import analyse_scan
from Experiments.ExploreFiltering import explore_filtering
from Experiments.FHCdiagrams import printFHCdiagrams, printISBIdiagrams
from Experiments.Visualise_mask import visualise_mask
from Experiments.data_together import explore_image_segmentation
from Experiments.loadingSavedModel import explore_reloading_models
from Experiments.ExploreFHC import explore_FHC
from Experiments.Unaceptable_analysis import unacceptable_analysis
from Experiments.measurements import measure_scan
from Experiments.Show_augmentations import show_augmentations
from Constants import *
from Experiments.make_many_models import train_all_models
from Experiments.FHC_measurements import fhc_all_models

scan_number = "A089R"

# analyse_scan(scan_number)

# explore_filtering(scan_number)

# visualise_mask("A004R")

# explore_image_segmentation()  # creates trains and saves a model

# explore_reloading_models()

# explore_FHC("A047L")

# unacceptable_analysis()

# measure_scan()

# show_augmentations("A099R")

# train_all_models()  # creates trains and saves all required models

fhc_all_models()

# printFHCdiagrams(scan_number)

# printISBIdiagrams(scan_number) # ISBI FIG2:scan_number="A089R"

print("complete")
