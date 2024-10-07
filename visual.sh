#!/bin/bash
 
cd pic
 
export GST_DEBUG_DUMP_DOT_DIR=pic/
 
declare -a filenames=("pipeline-media-type" "pipeline-caps-details" "pipeline-non-default-params" "pipeline-states" "pipeline-full-params" "pipeline-all" "pipeline-verbose")
 
for filename in "${filenames[@]}"
do
  # dot -Tpdf "${filename}.dot" > "${filename}.pdf"
  # dot -Tjpg "${filename}.dot" > "${filename}.jpg"
  dot -Tpng "${filename}.dot" > "${filename}.png"
done