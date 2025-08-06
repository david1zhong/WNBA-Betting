#!/bin/bash
while getopts s:e:r: flag
do
    case "${flag}" in
        s) START_YEAR=${OPTARG};;
        e) END_YEAR=${OPTARG};;
        r) RESCRAPE=${OPTARG};;
    esac
done

for i in $(seq "${START_YEAR}" "${END_YEAR}")
do
    echo "$i"
    git pull >> /dev/null
    git config --local user.email "action@github.com"
    git config --local user.name "Github Action"

    Rscript R/espn_wnba_03_player_box_creation.R -s $i -e $i

    if [ -f "wnba/player_box/csv/player_box_${i}.csv.gz" ]; then
        gzip -d -f "wnba/player_box/csv/player_box_${i}.csv.gz"
    fi

    git add "wnba/player_box/csv/player_box_${i}.csv" >> /dev/null
    git commit -m "PlayerBox CSV Update (Year: $i)" >> /dev/null || echo "No changes to commit"
    git pull --rebase >> /dev/null
    git push >> /dev/null
done
