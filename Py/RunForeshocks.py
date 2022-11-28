from Foreshocks_Aftershocks_productivity import *
# Cat_names = ['INGV2', 'SCEDC', 'NCEDC', 'OVSICORI', 'JMA', 'ALASKA2', 'KOERI2', 'NOA'] #  bar figure
# Cat_names = ['NEIC', 'GCMT2']
Cat_names = ['NEIC'] # AUOT
# Cat_names = ['CR']
work_folder = '/Users/nadavwetzler/Library/CloudStorage/GoogleDrive-nadav.wetzler@gmail.com/My Drive/DataSet/UCSC/Foreshocks/'
# Cmethod = 'WnC'
# Cmethod = 'ZnBZ'
Cmethod = 'TMC'
# Cmethod = 'GnK'

MCP = 5.0  # Constant Mc for all cats. Use -999 if loaded from csv
ddays = 30
dM = 0.0
max_depth = 70
dmSwarm = 2.0  # minimum aperture between M0 and largest fs / as
minN4b = 30  # minimum number of earthquakes to calculate b-value
nbs = 500  # number of bootstraps for productivity
LoadClusterData = True  # Use false to remake clusters, make sure
LoadEQData = True
Load_b_valueData = True


def main():
    print('Running Foreshocks!!!')
    # Get catalogs
    DATA = CatData(Cat_names, work_folder, ddays, dM, minN4b, dmSwarm, nbs, max_depth, MCP)

    # trim data spatially
    DATA.trimcats()

    # save cat text files for TG Zaliapin run
    # DATA.MakeCattxt4TG(work_folder + 'Catalogs4zaliapin')
    
    # DATA.Cat2ZMAP(work_folder + 'Catalogs4zaliapin')

    # calc b-value and Mc for each catalog
    DATA.CalcGR_Mc_cats(Load_b_valueData, save_b=False)

    # remove eq beloc completness level
    DATA.AboveMcCat(Mc=MCP)

    # make clusters for all selected clusters and calc b-values
    DATA.makeClusters(Cmethod, LoadClusterData)
    
    # clac max foreshock
    # DATA.calcRclust()
    
    # DATA.calaAreaSeis(0.5)

    # DATA.calc_k_clusters(plotter=False)

    # DATA.calc_mainshocks_rates(loader=True)

    # DATA.InterIntra(loader=True)
    
    # DATA.calc_fs_moment_ratios()

    # ----------------PLOTTING-----------------------------
    # DATA.Plotter('Globalregions')  # Regional - Figure 1
    # DATA.Plotter('MapSeis3')  # Global - Figure 1
    # DATA.Plotter('ShowIndividualClusters')  # Does not work
    # DATA.Plotter('MapCatClusters')
    # DATA.Plotter('b-values')
    # DATA.Plotter('Poissonian')  # Regional  - Figure S10
    # DATA.Plotter('c_typePie')  # Regional+Global - Fig S3, Fig S2
    # DATA.Plotter('c_type_map')  # Does not work
    # DATA.Plotter('ForeshocksDuration')
    # DATA.Plotter('dM_M')
    # DATA.Plotter('Aftershocks_b_value_yes_no_fs')
    # DATA.Plotter('Foreshocks_Bath')  # Regional - Fig. S8
    # DATA.Plotter('Foreshocks_b_value')  # Regional+Global - Fig 4a, Fig.5a
    # DATA.Plotter('Foreshocks_b_value-1a')  # Regional+Global - Fig 4b, Fig.5b
    # DATA.Plotter('Foreshocks_b_value-1a_EW')
    # DATA.Plotter('Foreshocks_b_value-1a_equal_t')
    # DATA.Plotter('Foreshocks_b_value_FMS')
    # DATA.Plotter('AftershockProductivity')
    # DATA.Plotter('ForeshockProductivity_c_type')
    # DATA.Plotter('BG_timeline_rates')
    # DATA.Plotter('AftershockProductivity_types_c_aperture')
    # DATA.Plotter('M0M1-productivity')
    # DATA.Plotter('AftershockWithForeshocksProductivity')
    # DATA.Plotter('AftershockWithForeshocksProductivity_c_type')
    # DATA.Plotter('Productivity_Rmax_Rpredicted')
    # DATA.Plotter('PlotProductivity_timing')
    # DATA.Plotter('AftershockWithForeshocksProductivity_c_type_aperture')
    # DATA.Plotter('AftershockProductivity_c_type')
    # DATA.Plotter('c_type_not_defined')
    # DATA.Plotter('Cat_poissonian')
    # DATA.Plotter('Map_AftershockWithForeshocksProductivity')
    # DATA.Plotter('Map_M0M1')
    # DATA.Plotter('b-val-productivity')
    # DATA.Plotter('foreshocks-Pie')
    # DATA.Plotter('foreshocks-Pie60d')
    # DATA.Plotter('foreshocks-Pie60dapr')
    # DATA.Plotter('FMS-Pie')
    DATA.Plotter('EW_Pacific')  # Global
    # DATA.Plotter('K_mainshocks')
    # DATA.Plotter('foreshocks_times')
    # DATA.Plotter('productive_foreshocks')
    # DATA.Plotter('foreshock_map_fms')
    # DATA.Plotter('plot_fs_moment_ratios')
    # DATA.Plotter('Foreshocks_Omori_EW')
    ##  DATA.Plotter('FMS_spindles') # not normalized
    # DATA.Plotter('FMS_spindles2') # normalized by Reff
    ## DATA.Plotter('FMS_spindles3') # normalized by Reff

    print('Done!!!')
    plb.show()


if __name__ == "__main__":
    main()
