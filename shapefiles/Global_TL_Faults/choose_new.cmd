bf_year=1976
bf_month=1
bf_day=1
year=2010
month=2
day=27
af_year=2015
af_month=4
af_day=28

lo1=`awk 'NR==1 {split($0,a,"="); printf "%.3f",a[2]} ' lon_lat_range.lst`
lo2=`awk 'NR==2 {split($0,a,"="); printf "%.3f",a[2]} ' lon_lat_range.lst`
la1=`awk 'NR==3 {split($0,a,"="); printf "%.3f",a[2]} ' lon_lat_range.lst`
la2=`awk 'NR==4 {split($0,a,"="); printf "%.3f",a[2]} ' lon_lat_range.lst`

echo $lo1 $lo2 $la1 $la2

awk -v year=$year -v month=$month -v day=$day -v lo1=$lo1 -v lo2=$lo2 -v la1=$la1 -v la2=$la2 ' {split($1,a,"/");
                       if( (a[1]<year || (a[1]==year && a[2]<month) || (a[1]==year && a[2]==month && a[3]<day)) && $4<=lo2 && $4>=lo1 && $3<=la2 && $3>=la1 ) 
                       print $0}' ../extracted.lst >try_bf.txt

awk -v year=$year -v month=$month -v day=$day -v lo1=$lo1 -v lo2=$lo2 -v la1=$la1 -v la2=$la2 '{split($1,a,"/");
                       if( (a[1]>year || (a[1]==year && a[2]>month) || (a[1]==year && a[2]==month && a[3]>=day)) && $4<=lo2 && $4>=lo1 && $3<=la2 && $3>=la1 )
                       print $0}' ../extracted.lst >try_af.txt

cat try_af.txt | gawk ' BEGIN{"cat main.txt"|getline;str1=$3;dip1=$4;rak1=$5;str2=$6;dip2=$7;rak2=$8;}  ($13>=str1-50 && $13<=str1+50 && $14>=dip1-20 && $14<=dip1+20  && $15>=rak1-30 && $15<=rak1+30) || ($16>=str1-50 && $16<=str1+50 && $17>=dip1-30 && $17<=dip1+30  && $18>=rak1-30 && $18<=rak1+30) || ($16>=str2-50 && $16<=str2+50 && $17>=dip2-30 && $17<=dip2+30 && $18>=rak2-30 && $18<=rak2+30) || ($13>=str2-50 && $13<=str2+50 && $14>=dip2-30 && $14<=dip2+30 && $15>=rak2-30 && $15<=rak2+30) || ($16>=str2-30 && $16<=str2+30 &&  $17>80 && $17<=90 && $18>-100 && $18<-80) || ($13>=str2-30 && $13<=str2+30 &&  $14>80 && $14<=90 && $15>-100 && $15<-80) || ($16>=str1-30 && $16<=str1+30 &&  $17>80 && $17<=90 && $18>-100 && $18<-80) || ($13>=str1-30 && $13<=str1+30 &&  $14>80 && $14<=90 && $15>-100 && $15<-80){print $0} '> try_af_thrust.txt

cat try_af.txt try_af_thrust.txt| sort | uniq -u > try_af_nonthrust.txt
cat try_af_thrust.txt | gawk '{print $4,$3,$5,$7,$8,$9,$10,$11,$12,$6,"X Y"}'> thrust_af
cat try_af_nonthrust.txt | gawk '{print $4,$3,$5,$7,$8,$9,$10,$11,$12,$6,"X Y"}'> nonthrust_af

#rm try_af.txt try_af_thrust.txt try_af_nonthrust.txt



cat try_bf.txt | gawk ' BEGIN{"cat main.txt"|getline;str1=$3;dip1=$4;rak1=$5;str2=$6;dip2=$7;rak2=$8;}  ($13>=str1-50 && $13<=str1+50 && $14>=dip1-20 && $14<=dip1+20  && $15>=rak1-30 && $15<=rak1+30) || ($16>=str1-50 && $16<=str1+50 && $17>=dip1-30 && $17<=dip1+30  && $18>=rak1-30 && $18<=rak1+30) || ($16>=str2-50 && $16<=str2+50 && $17>=dip2-30 && $17<=dip2+30 && $18>=rak2-30 && $18<=rak2+30) || ($13>=str2-50 && $13<=str2+50 && $14>=dip2-30 && $14<=dip2+30 && $15>=rak2-30 && $15<=rak2+30) || ($16>=str2-30 && $16<=str2+30 &&  $17>80 && $17<=90 && $18>-100 && $18<-80) || ($13>=str2-30 && $13<=str2+30 &&  $14>80 && $14<=90 && $15>-100 && $15<-80) || ($16>=str1-30 && $16<=str1+30 &&  $17>80 && $17<=90 && $18>-100 && $18<-80) || ($13>=str1-30 && $13<=str1+30 &&  $14>80 && $14<=90 && $15>-100 && $15<-80){print $0} '> try_bf_thrust.txt

cat try_bf.txt try_bf_thrust.txt| sort | uniq -u > try_bf_nonthrust.txt
cat try_bf_thrust.txt | gawk '{print $4,$3,$5,$7,$8,$9,$10,$11,$12,$6,"X Y"}'> thrust_bf
cat try_bf_nonthrust.txt | gawk '{print $4,$3,$5,$7,$8,$9,$10,$11,$12,$6,"X Y"}'> nonthrust_bf

#rm try_bf.txt try_bf_thrust.txt try_bf_nonthrust.txt

