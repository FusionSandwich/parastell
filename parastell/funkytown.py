import parastell.magnet_coils as magnet_coils

coils_file = "wout_vmec.nc"
rect_cross_section = ["rectangle", 20, 60]
toroidal_extent = 90.0

rect_coil_obj = magnet_coils.MagnetSet(
    coils_file, rect_cross_section, toroidal_extent
)
