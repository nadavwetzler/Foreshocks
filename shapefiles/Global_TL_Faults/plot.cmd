gmtset COLOR_NAN=-
gmtset FRAME_PEN=0.3p
gmtset FRAME_WIDTH=0.05i

out=gcmt_normal.ps
input=gcmtcmt_normal
lo1=139
lo2=145
la1=34
la2=41
ar_lo1=
ar_la1=
ar_lo2=
ar_la2=
ar_lo3=
ar_la3=
ar_lo4=
ar_la4=
grdcut ~/Japan_moment/Other/codes/topo1.grd -R139/145/34/41 -Gsc.grd
makecpt  ~/Japan_moment/Other/codes/-CETOPO1 -T-11000/9000/500 -Z > sc.cpt
grdgradient  sc.grd -Gintense.grd -A10 -Nt
grdimage sc.grd -JM6i -R$lo1/$lo2/$la1/$la2 -B1/1:.GCMT2011~2013:WSEN  -Iintense.grd -Csc.cpt -P -K > $out

pscoast -JM6i -R$lo1/$lo2/$la1/$la2 -B -W2p  -Df -Ia  -Na/5/0/255/0 -K -O -P>>$out
psxy -R -JM -M ~/Japan_moment/codes/trench.txt -W2p/255/0/255 -B -O -K -P >> $out
psmeca $input -R -JM -Ewhite -B -Gred  -L1p  -W1p -Sm0.6/-0.6  -T0  -O -K -P >> $out
#psmeca maincmt -R -JM -D0/800 -E255/255/255 -B -G180/0/0 -Sm0.6/-0.6  -T0  -O -K -P >> $out
psxy -R -JM -W2pta -B -K -O -P<<END>>$out
$ar_lo1 $ar_la1
$ar_lo2 $ar_la2
$ar_lo3 $ar_la3
$ar_lo4 $ar_la4
$ar_lo1 $ar_la1
END
ps2pdf $out
