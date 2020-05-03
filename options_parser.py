import argparse


def setup_options():
    options = argparse.ArgumentParser()
    options.add_argument('-ci', action='store', dest='cell_info',
                         default='GSE92742_Broad_LINCS_cell_info.txt')
    options.add_argument('-gi', action='store', dest='gene_info',
                             default='GSE92742_Broad_LINCS_gene_info.txt')
    options.add_argument('-ii', action='store', dest='inst_info',
                             default='GSE92742_Broad_LINCS_inst_info.txt')
    options.add_argument('-pi', action='store', dest='pert_info',
                             default='GSE92742_Broad_LINCS_pert_info.txt')
    options.add_argument('-d', action='store', dest='data',
                         default='GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx')
    return options.parse_args()
