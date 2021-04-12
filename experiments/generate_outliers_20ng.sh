#!/bin/bash
cat ./datasets/20ng_od/test/misc.txt ./datasets/20ng_od/test/pol.txt ./datasets/20ng_od/test/rec.txt ./datasets/20ng_od/test/rel.txt ./datasets/20ng_od/test/sci.txt > ./datasets/20ng_od/test/comp-outliers.txt
cat ./datasets/20ng_od/test/comp.txt ./datasets/20ng_od/test/pol.txt ./datasets/20ng_od/test/rec.txt ./datasets/20ng_od/test/rel.txt ./datasets/20ng_od/test/sci.txt > ./datasets/20ng_od/test/misc-outliers.txt
cat ./datasets/20ng_od/test/comp.txt ./datasets/20ng_od/test/misc.txt ./datasets/20ng_od/test/rec.txt ./datasets/20ng_od/test/rel.txt ./datasets/20ng_od/test/sci.txt > ./datasets/20ng_od/test/pol-outliers.txt
cat ./datasets/20ng_od/test/comp.txt ./datasets/20ng_od/test/misc.txt ./datasets/20ng_od/test/pol.txt ./datasets/20ng_od/test/rel.txt ./datasets/20ng_od/test/sci.txt > ./datasets/20ng_od/test/rec-outliers.txt
cat ./datasets/20ng_od/test/comp.txt ./datasets/20ng_od/test/misc.txt ./datasets/20ng_od/test/pol.txt ./datasets/20ng_od/test/rec.txt ./datasets/20ng_od/test/sci.txt > ./datasets/20ng_od/test/rel-outliers.txt
cat ./datasets/20ng_od/test/comp.txt ./datasets/20ng_od/test/misc.txt ./datasets/20ng_od/test/pol.txt ./datasets/20ng_od/test/rec.txt ./datasets/20ng_od/test/rel.txt > ./datasets/20ng_od/test/sci-outliers.txt
