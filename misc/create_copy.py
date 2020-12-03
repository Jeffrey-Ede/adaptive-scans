import os
from shutil import copy

SOURCE = f"//ads.warwick.ac.uk/shared/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/recurrent_conv-1/"
SINK = "//Desktop-sa1evjv/h/sink/recurrent_conv-1/"

WHITELIST = [str(i) for i in range(98, 144)]

for item in os.listdir(SOURCE):
    path = SOURCE+item
    if os.path.isdir(item) and item in WHITELIST:
        os.makedirs(SINK+item)
        for subitem in os.listdir(path):
            subpath = path + "/" + subitem
            dst = SINK + item + "/" + subitem
            if os.path.isdir(subpath):
                os.makedirs(dst)
                if subitem == "dnc":
                    for file in os.listdir(subpath):
                        if not file == "__pycache__":
                            source_loc = subpath + "/" + file
                            sink_loc = dst + "/" + file
                            copy(source_loc, sink_loc)
            elif subitem.split(".")[-1] in ["py"] or subitem == "notes.txt":
                copy(subpath, dst)

    elif os.path.isfile(path):
        try:
            copy(path, SINK+item)
        except:
            continue