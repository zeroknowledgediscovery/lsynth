for all of the ten datasets they were run with this command on mcc 

./train_qnet_fractions.py --data datasets/{dataset}.csv --outdir data/gss_2024 --prefix {dataset} --fractions 0.075,0.1,0.125,0.150,0.175,0.200,0.225,0.250,0.275,0.300,0.325,0.350,0.375,0.400,0.425,0.450,0.475,0.500,0.525,0.550,0.575,0.600,0.625,0.650,0.675,0.700,0.725,0.750,0.775,0.800,0.825,0.850 --n_jobs 15 