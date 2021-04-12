#!/bin/bash
cat ./datasets/ag_od/test/sci.txt ./datasets/ag_od/test/sports.txt ./datasets/ag_od/test/world.txt > ./datasets/ag_od/test/business-outliers.txt
cat ./datasets/ag_od/test/business.txt ./datasets/ag_od/test/sports.txt ./datasets/ag_od/test/world.txt > ./datasets/ag_od/test/sci-outliers.txt
cat ./datasets/ag_od/test/business.txt ./datasets/ag_od/test/sci.txt ./datasets/ag_od/test/world.txt > ./datasets/ag_od/test/sports-outliers.txt
cat ./datasets/ag_od/test/business.txt ./datasets/ag_od/test/sci.txt ./datasets/ag_od/test/sports.txt > ./datasets/ag_od/test/world-outliers.txt