wget https://github.com/xg-chu/GPAvatar/releases/download/Resources/resources.tar -O ./resources.tar

# check file integrity
md5sum ./resources.tar
printf "1d386517baa670307243c3fd45494a53 Please check if the md5sum is correct\n"

tar -xvf resources.tar

mv resources/examples ./demos/
mv resources/drivers ./demos/

mv resources/main_params/* ./core/libs/FLAME/assets/
mv resources/track_params/* ./core/libs/lightning_track/engines/FLAME/assets/
mv resources/matting/* ./core/libs/lightning_track/engines/human_matting/assets/
mv resources/emoca/* ./core/libs/lightning_track/engines/emoca/assets/
mv resources/mica_base/* ./core/libs/lightning_track/engines/mica/assets/

rm -rf resources/
rm -rf resources.tar
