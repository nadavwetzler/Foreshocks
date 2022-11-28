gmtset COLOR_BACKGROUND=white
gmtset COLOR_NAN=-
gmtset FRAME_PEN=0.3p
gmtset FRAME_WIDTH=0.05i
x1=tmp5
y1=tmp6
out=gcmt_cumu2.ps
input=gcmt_cumu2.txt
center=21
makecpt -C/Workspace/Japan_moment/after2011/thrust/moment/mat/Reds_09.cpt  -T20/28/0.2 > 1.cpt
makecpt -C/Workspace/Japan_moment/after2011/thrust/moment/mat/Reds_09.cpt  -T0/3200/20 > 3.cpt
#topography
#grdcut topo1.grd -R139/145/34/41 -Gsc.grd
#makecpt  -CETOPO1 -T-11000/9000/500 -Z > sc.cpt
#grdimage sc.grd -JM6i -R139/145/34/41 -B  -C1.cpt -P  -K > $out
xyz2grd $input  -F -R139/145/34/41 -I0.3 -G1.grd
xyz2grd slip.txt  -F -R139/145/34/41 -I0.3 -G3.grd
grdsample 1.grd -G2.grd -I0.1 -F -R139/145/34/41
grdsample 3.grd -G4.grd -I0.1 -F -R139/145/34/41
#grdimage 4.grd -JM6i -R139/145/34/41 -B  -C3.cpt -P  -K > $out
grdimage 2.grd -JM6i -R139/145/34/41 -B  -C1.cpt -P  -K > $out
#moment
#psxy $input -JM6i -R139/145/34/41 -Ss0.12i -B -C1.cpt -P -K>$out
#pscontour -JM -R -B -C1.cpt -P -K -W1p -O>>$out
#pscontour -JM -R -B -C1.cpt -P -K  -I -X3i -O>>$out
#pscoast
#psmeca $input -R -JM -D0/800 -E255/255/255 -G255/0/0 -Sm0.6/-0.6  -T0  -O -K -P >> $out
#psmeca maincmt -R -JM -D0/800 -E255/255/255 -G180/0/0 -Sm0.6/-0.6  -T0  -O -K -P >> $out
grdcontour 2.grd  -JM -A4apf10  -G40 -L20/28 -Wcthinnest -P -O -K>>$out
grdcontour 3.grd  -JM -A1000f10greds15 -C500 -Wa6/0/100/255 -Wc3/0/200/255 -G200 -L0/3200 -P -O -K>>$out
pscoast -JM6i -R139/145/34/41 -W1p -B1f0.5/1f0.5 -Df -Ia -Na/5/0/255/0 -K -P -O>>$out
psxy -R -JM -W2pta -O -K -P<<END>>$out
142.6 35.39
140.218 36.032
142.218 41.143
144.6 40.49
142.6 35.39
END
#pswiggle $x1 -Z4 -B -R -JM -K -A270 -G240/220/220  -C$center -Wthinner -O  >>$out
#pswiggle $y1 -Z4 -B -R -JM -K -A90 -G240/220/220  -C$center -Wthinner -O  >>$out
psxy -R -JM -M /Workspace/Japan_moment/after2011/thrust/moment/mat/trench.txt -W1pta/0/0/0 -O -K -P >> $out
pstext -R -JM -K -O -P<<END >> $out
144.6 40.5 10 0 10 BL 20
END
psscale -D6.5i/2i/8c/0.3c -C1.cpt -I -E -B1::/:exp:  -K -O >> $out
ps2pdf $out
