#!/bin/bash

#  IN: repo.txt \w list of repo urls
# OUT: /repos/...

# GNU awk is required for Two-Way coprocess commication, to get file size
# http://www.gnu.org/software/gawk/manual/html_node/Two_002dway-I_002fO.html#Two_002dway-I_002fO
command -v gawk >/dev/null 2>&1 || { echo "Please install GNU awk" >&2; exit 1; }


mkdir -p repos
pushd repos

cat ../repos.txt | xargs -L1 git clone --depth 1

for D in *; do
    if [ -d "${D}" ]; then
        echo "Analyzing ${D}"
        cd "${D}"

        linguist --json | \
            jq -r 'keys[] as $k | "\($k);\(.[$k][])"' | \
            gawk -F';' -v pwd="$PWD" '{path = pwd"/"$2; wc = "wc -l <"path; wc |& getline size; close(wc); print $1 ";" path ";" size}' > files.csv

        echo "${D}" > status.txt
        linguist   >> status.txt
        cd ..
    fi
done

find . -name "files.csv" -type f -print0 | xargs -0 cat | sort > ../files_all.csv

find . -name "status.txt" -type f -print0 | xargs -0 cat > ../status_all.txt

popd
