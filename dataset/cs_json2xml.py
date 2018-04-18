#!/usr/bin/env python

import xml
import xml.dom.minidom
import json
from xml.dom.minidom import getDOMImplementation
import xml.etree.ElementTree as et
import sys
import os

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = et.tostring(elem, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml()

def main():
    json_fname = sys.argv[1] # "aachen_000000_000019_gtFine_polygons.json"
    xml_fname = json_fname.replace(".json",".xml")

    parsed = json.loads(open(json_fname).read())

    top = et.Element('annotation')
    filename = et.SubElement(top,"filename")
    filename.text = os.path.basename(json_fname.replace("json","jpg")).replace("gtFine_polygons","leftImg8bit")
    folder = et.SubElement(top,"folder")
    folder.text = "cityscapes"
    source = et.SubElement(top,"source")
    sourceImage = et.SubElement(source,"sourceImage")
    sourceImage.text = "The MIT-CSAIL database of objects and scenes"
    sourceAnnotation = et.SubElement(source,"sourceAnnotation")
    sourceAnnotation.text = "LabelMe Webtool"
    imagesize = et.SubElement(top,"size")
    nrows = et.SubElement(imagesize,"height")
    ncols = et.SubElement(imagesize,"width")
    depth = et.SubElement(imagesize,"depth")
    nrows.text, ncols.text, depth.text = str(int(round(parsed["imgHeight"]/2))), str(int(round(parsed["imgWidth"]/2))), '3'  # SCALE TO HALF SIZE HERE !!!!

    objects = parsed["objects"]
    idval = 0

    for idval,label in enumerate(objects):
        cls_label = label["label"]
        polygons = label["polygon"]

        obj = et.SubElement(top,"object")
        name = et.SubElement(obj,"name")
        name.text = cls_label
        deleted = et.SubElement(obj,"deleted")
        deleted.text = "0"
        difficult = et.SubElement(obj,"difficult")
        difficult.text = "0"
        verified = et.SubElement(obj,"verified")
        verified.text = "0"
        occluded = et.SubElement(obj,"occluded")
        occluded.text = "no"
        date = et.SubElement(obj,"date")
        date.text = "31-May-2017 06:26:28"
        id_tag = et.SubElement(obj,"id")
        id_tag.text = str(idval)
        polygon = et.SubElement(obj,"polygon")
        username = et.SubElement(polygon,"username")
        username.text = "admin"
        xminval, xmaxval, yminval, ymaxval = ncols, 0, nrows, 0
        for p in polygons:
            p[0], p[1] = int(round(p[0]/2)), int(round(p[1]/2)) # SCALE TO HALF SIZE HERE !!!!
            pt = et.SubElement(polygon,"pt")
            x = et.SubElement(pt,"x")
            y = et.SubElement(pt,"y")
            x.text, y.text = str(p[0]), str(p[1])
            xminval = min(xminval, p[0])
            xmaxval = max(xmaxval, p[0])
            yminval = min(yminval, p[1])
            ymaxval = max(ymaxval, p[1])
        bndbox = et.SubElement(obj,"bndbox")
        xmin = et.SubElement(bndbox,"xmin")
        xmax = et.SubElement(bndbox,"xmax")
        ymin = et.SubElement(bndbox,"ymin")
        ymax = et.SubElement(bndbox,"ymax")
        xmin.text, xmax.text = str(xminval), str(xmaxval)
        ymin.text, ymax.text = str(yminval), str(ymaxval)


    pretty_xml_as_string = prettify(top)
    pretty_json_as_string = json.dumps(parsed, indent=2, sort_keys=True)

    # print pretty_xml_as_string

    with open(xml_fname,"wt") as fp:
        fp.write(pretty_xml_as_string)


if __name__=="__main__":
    main()
