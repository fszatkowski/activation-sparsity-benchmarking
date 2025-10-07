I tried to follow harness as much as possible, but this repo still requires some tweaks. The user of any scripts should set the following environment vars to make sure transformers will have access to all the models you might want to use, and the models and datasets will be downloaded to the desired directory:
* HT_TOKEN
* HF_HOME
Additionally, when using slurm, you should set:
* SLURM_ACC
* SLURM_PARTITION
