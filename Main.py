from Experiments.AnalyseScan import analyse_scan
from Experiments.ExploreFiltering import explore_filtering
from Experiments.Visualise_mask import visualise_mask
from Experiments.data_together import test_data

scan_number = "A080L"

analyse_scan(scan_number)

explore_filtering(scan_number)

visualise_mask("A001R")

test_data()

print("complete")
