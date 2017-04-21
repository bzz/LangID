#!/bin/bash

#  IN: repo.txt \w list of repo urls
# OUT: 

#cat repos.txt | xargs -L1 git clone --depth 1

for D in *; do
    if [ -d "${D}" ]; then
        echo "${D}"
        cd "${D}"
        linguist --json | jq -r 'keys[] as $k | "\($k);\(.[$k][])"' | awk -F';' -v d="$PWD" '{print $1 ";" d "/" $2}' > files.csv
        cd ..
    fi
done

find . -name "files.csv" -type f -print0 | xargs -0 cat | sort > files_all.csv

