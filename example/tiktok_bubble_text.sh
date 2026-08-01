#!/bin/bash
font="$HOME/Library/Fonts/DejaVuSans.ttf"
pointsize=60
radius=5
vpad=4
overlap=10
maxwidth=500
text="Hello jorld another text here very long text text 2 1234"

spacew=$(( $(magick -background white -fill black -font "$font" \
  -pointsize $pointsize label:"X " -format "%w" info:) \
  - $(magick -background white -fill black -font "$font" \
  -pointsize $pointsize label:"X" -format "%w" info:) ))
pad=$((radius + spacew))

read -ra words <<< "$text"

lines=()
line=""
for word in "${words[@]}"; do
  candidate="$line $word"
  candidate="${candidate# }"
  width=$(magick -background white -fill white -font "$font" \
    -pointsize $pointsize label:"$candidate" \
    -bordercolor white -border ${pad}x${radius} -format "%w" info:)
  if [ -n "$line" ] && [ "$width" -gt "$maxwidth" ]; then
    lines+=("$line")
    line="$word"
  else
    line="$candidate"
  fi
done
lines+=("$line")

tmp=$(mktemp -d)
maxw=0
heights=()
i=0
for line in "${lines[@]}"; do
  magick -background white -fill black -font "$font" -pointsize $pointsize \
    label:"$line" -bordercolor white -border ${pad}x${vpad} \
    \( +clone -alpha extract \
       -fill black -stroke none \
       -draw "roundrectangle 0,0 %[fx:w-1],%[fx:h-1] $radius,$radius" \
       -negate -alpha off \) \
    -compose CopyOpacity -composite \
    "$tmp/$(printf '%02d' $i).png"
  width=$(magick "$tmp/$(printf '%02d' $i).png" -format "%w" info:)
  [ "$width" -gt "$maxw" ] && maxw=$width
  heights+=("$(magick "$tmp/$(printf '%02d' $i).png" -format "%h" info:)")
  i=$((i + 1))
done

for f in "$tmp"/*.png; do
  magick "$f" -background none -gravity center -extent ${maxw}x \
    "$f"
done

files=("$tmp"/*.png)
n=${#files[@]}
total=0
for h in "${heights[@]}"; do total=$((total + h)); done
total=$((total - (n - 1) * overlap))
magick -size ${maxw}x${total} xc:none "$tmp/canvas.png"
y=0
for i in $(seq 0 $((n - 1))); do
  magick "$tmp/canvas.png" "${files[$i]}" \
    -compose darken -geometry +0+${y} -composite "$tmp/canvas.png"
  y=$((y + ${heights[$i]} - overlap))
done
magick "$tmp/canvas.png" output.png
rm -rf "$tmp"
