#!/bin/bash
# Source: https://github.com/cvdfoundation/kinetics-dataset

# Splits: train, val, test
split=${1}

# Download directories vars
root_dl=${2:-/workdir/data/kinetics_400}
root_dl_targz="${root_dl}_targz"

echo -e "\nDownload $split kinetics400 to '$root_dl'"

# Make root directories
[ ! -d "$root_dl" ] && mkdir "$root_dl"
[ ! -d "$root_dl_targz" ] && mkdir "$root_dl_targz"

# Download tars, will resume
curr_dl="${root_dl_targz}/${split}"
url="https://s3.amazonaws.com/kinetics/400/${split}/k400_${split}_path.txt"
[ ! -d "$curr_dl" ] && mkdir -p "$curr_dl"
wget -c -i "$url" -P "$curr_dl"

# Download annotations csv files
curr_dl=${root_dl}/annotations
url="https://s3.amazonaws.com/kinetics/400/annotations/${split}.csv"
[ ! -d "$curr_dl" ] && mkdir -p "$curr_dl"
wget -c "$url" -P "$curr_dl"

# Downloads complete
echo -e "\nDownloads complete!"

# Extract
curr_dl="${root_dl_targz}/${split}"
curr_extract="${root_dl}/${split}"
[ ! -d "$curr_extract" ] && mkdir -p "$curr_extract"
tar_list=$(ls "$curr_dl")
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting "$curr_dl/$f" to "$curr_extract" && tar zxf "$curr_dl/$f" -C "$curr_extract"
done

# Extraction complete
echo -e "\nExtractions complete!"
