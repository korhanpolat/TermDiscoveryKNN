

explanation regarding paths, all paths must be absolute paths.

Filenames for signers and cross-validation groupds are stored in this directory
`"CVroot": "{your_directory}/knn_utd/data/CVfolds"`

where to save scores and discovered objects
`"exp_root": "{your_directory}/results/"`

location of the features
`"feats_root": "{your_directory}/features/"`


for `"eval"` :
bash script that calls TDE
`"tderunfile":"{your_directory}/knn_utd/run_tde.sh"`
where you've built TDE
`"TDEROOT":"{your_TDE_directory}/tdev2/tdev2"`
main conda source
`"TDESOURCE":"{your_conda_directory}/etc/profile.d/conda.sh"`
Config file that is used for to Phoenix dataset
`"config_file": "{your_directory}/knn_utd/config/config_phoenix.json"`

for tuning:
where to save `skopt.Tuner` checkpoints
`"chkpoint_root":"{your_directory}/knn_utd/data/checkpoints"`

