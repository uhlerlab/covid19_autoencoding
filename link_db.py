import data_loader as dl
import options_parser as op
import pandas as pd
import pickle as p


# This is just for me to cache a bunch of fields via pickle
# I didn't expect all these fields to be necessary so I just hard
# coded the look ups, but we can easily modify this
def main(args):
    inst_info = pd.read_csv(args.inst_info, sep='\t')
    pairs = dl.get_all_pairs(args.data)
    keys = [p[0] for p in pairs]
    rows = inst_info['inst_id'].isin(keys) == True
    key = list(inst_info[rows]['inst_id'])
    vals = list(inst_info[rows]['cell_id'])
    pert_id = list(inst_info[rows]['pert_iname'])
    pert_dose = list(inst_info[rows]['pert_dose'])
    pert_time = list(inst_info[rows]['pert_time'])
    pert_type = list(inst_info[rows]['pert_type'])
    rna_plate = list(inst_info[rows]['rna_plate'])
    rna_well = list(inst_info[rows]['rna_well'])

    out = {key[i]: vals[i] for i in range(len(key))}
    p.dump(out, open('pert_cell_map.p', 'wb'))
    pert_out = {key[i]: (pert_dose[i], pert_time[i], pert_id[i], pert_type[i],
                         rna_plate[i], rna_well[i]) \
                for i in range(len(key))}
    p.dump(pert_out, open('pert_dose_map.p', 'wb'))

if __name__ == "__main__":
    args = op.setup_options()
    main(args)
