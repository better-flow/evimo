import math
VALS = {
        "flea": {"res": (1552, 2080), "f" :(2062.63, 2062.48)},
        "right": {"res": (480, 640), "f" :(558.417,557.475)},
        "left": {"res": (480, 640), "f" :(556.184,555.632)},
        "mono": {"res": (480, 640), "f" :(519.638,519.384)}}

def hyp(lst):
    return math.sqrt(sum([i*i for i in lst]))

def mean(lst):
    return sum(lst)/len(lst)

for cam in VALS:
    print("Camera: %s"%cam)
    res = VALS[cam]["res"]
    print("\tRes  (in pixels): %s; Hypotenuse %f"%(str(res), hyp(res)))
    fs = VALS[cam]["f"]
    print("\tFoci (in pixels): %s; Mean: %f"%(str(fs), mean(fs)))
    fov = math.degrees(2.0 * math.atan(hyp(res)/(2.0*mean(fs))))
    print("\tFOV  (in degrees): %f"%fov)


