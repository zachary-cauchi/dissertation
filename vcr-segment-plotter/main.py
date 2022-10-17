from fileinput import filename
import os
import json
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy
from multiprocessing import Pool

random_generator = numpy.random.RandomState(1234)
baseDir = os.path.dirname(os.path.realpath(__file__))
resDir = os.path.join(baseDir, "res")
# resDir = os.path.join(baseDir, "../datasets/vcr/vcr1images")
destDir = os.path.join(baseDir, "out")

def get_random_color():
    return random_generator.rand(3,)

def get_jsons(folder):
    result = []
    for dirpath, dirname, filenames in os.walk(folder):
        result.extend([os.path.join(dirpath, filename) for filename in filenames if filename.endswith(".json")])
    return result

def process_json(jsonName):
    imageName, *_ = os.path.splitext(os.path.basename(jsonName))
    outputName = imageName + ".jpg"
    imageName += ".jpg"

    print("Processing file " + imageName)

    imageName = os.path.join(os.path.dirname(jsonName), imageName)

    file = json.load(open(jsonName))

    # Load the image and prepare to plot.
    img = plt.imread(imageName)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_axis_off()

    # For each segment entry, plot it on the image at 40% transparency.
    # Also show the legend for each segment and hide the axis.
    for i, segm in enumerate(file["segms"]):
        for subsegm in segm:
            poly = Polygon(subsegm)
            poly.set_facecolor(get_random_color())
            poly.set_alpha(0.4)
            poly.set_label(file["names"][i] + str(i + 1))
            ax.add_patch(poly)
            ax.legend()

    # Save the new image.
    plt.savefig(os.path.join(destDir, outputName))
    plt.close()

def process_jsons(jsons):
    # Disable plot windows to prevent constant popup.
    plt.ioff()
    
    # Process each json metadata file found using a thread pool.
    with Pool(processes= min(os.cpu_count(), 4, len(jsons))) as pool:
        pool.map(process_json, jsons)

jsons = get_jsons(resDir)
process_jsons(jsons)
