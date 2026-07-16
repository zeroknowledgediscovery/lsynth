#!/bin/bash

compiler=pdflatex
args=()

while [[ $# -gt 0 ]]
do
    case "$1" in
        -c)
            case "$2" in
                lua) compiler=lualatex ;;
                xe)  compiler=xelatex ;;
                *)
                    echo "Usage: $0 [-c lua|xe] texfile" >&2
                    exit 2
                    ;;
            esac
            shift 2
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

texfile=${args[0]}
inputs="${args[*]:1}"

while true
do
latexmk -f -e "\$pdflatex=q/$compiler %O --shell-escape %S/" -pdf "$texfile"
sleep 2
done
