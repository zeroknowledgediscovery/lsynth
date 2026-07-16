python lsynth_eval_synthpop_optional.py 2018 \
  --run-synthpop \
  --synthpop-train-rows 1000 \
  --synthpop-method cart \
  --synthpop-maxfaclevels 300

# for using all original dataframe
python lsynth_eval_synthpop_optional.py 2018 --run-synthpop --synthpop-train-rows 0
