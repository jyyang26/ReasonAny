# Setup
# Model Merging
## Safety Model
`cd safety`
`bash run_safety.sh`

## Biomedicine Model
`cd biomedicine`
`bash run_bio.sh`

## Finance Model
`cd finance`
`bash run_finance.sh`

# Hyperparameter
## Safety
`cd safety`
and switch the SAFETY_SCALE and REASONING_SCALE in run_safety.sh to set the $\lambda$,
switch the SAFETY_RATIO and REASONING_RATIO in run_safety.sh to set the selection ratio $p$.
`bash run_safety.sh`

## Biomedicine
`cd biomedicine`
and switch the DOMAIN_SCALE and REASONING_SCALE in run_bio.sh to set the $\lambda$,
switch the DOMAIN_RATIO and REASONING_RATIO in run_bio.sh to set the selection ratio $p$.
`bash run_bio.sh`

## Finance
`cd finance`
and switch the DOMAIN_SCALE and REASONING_SCALE in run_finance.sh to set the $\lambda$,
switch the DOMAIN_RATIO and REASONING_RATIO in run_finance.sh to set the selection ratio $p$.
`bash run_finance.sh`
