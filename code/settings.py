categories = ["P1", "P2"]

enlarge = 5 #150 Number of pixels to enlarge the regoion to improve tracking (increase number for large displacements)
overlap_to_join = 0.5 # [percentage]
window = 64 # detection window size [pixels]
half_window = window/2.0 # [pixels]

scale = 1.0 # image scaling
inverse_scale = 1.0/scale
score_threshold = 0.5 # consider regions above this confidence value.
max_detections = 1000

rgb = True
