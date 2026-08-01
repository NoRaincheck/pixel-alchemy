# target resolution: 1080 by 1920
# default bubble text
magick -size 700x200 xc:none \
  \( -size 620x110 -background none \
     -font "$HOME/Library/Fonts/DejaVuSans.ttf" -pointsize 52 -gravity center \
     -stroke black -strokewidth 10 -fill black caption:'Your text here' \
     -stroke none -fill white caption:'Your text here' \) \
  \( +clone -background black -shadow 60x4+0+4 \) \
  +swap -background none -layers merge +repage \
  -gravity center -extent 700x200 \
  tiktok_style.png