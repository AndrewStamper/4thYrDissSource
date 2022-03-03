"""
This is the main API which calls all subsequent actions
"""

from Experiments.AnalyseScan import analyse_scan
from Experiments.ExploreFiltering import explore_filtering
from Experiments.Visualise_mask import visualise_mask
from Experiments.data_together import explore_image_segmentation
from Experiments.loadingSavedModel import explore_reloading_models

scan_number = "A080L"

#analyse_scan(scan_number)

#explore_filtering(scan_number)

#visualise_mask("A001R")

#explore_image_segmentation()

explore_reloading_models()


print("complete")
