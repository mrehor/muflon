#!/bin/bash
#
#SBATCH --job-name=all_benchmarks
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --partition=express
#SBATCH --constraint="InfiniBand"
#SBATCH --mail-user=$USER@karlin.mff.cuni.cz
#SBATCH --mail-type=END,FAIL

# NOTE:
#   The above default values can be overwritten by specifying environment
#   variables (e.g. SBATCH_JOB_NAME="test_mms_Ia") and these can be further
#   overwritten by command line arguments (e.g. sbatch -n 4 <this_script>)

JOBNAME=${SLURM_JOB_NAME}

SCRIPTDIR=`pwd`
OUTPUTDIR="/srv/nobackup/$USER/muflon-results/test/bench"

echo Running ${JOBNAME} on host `hostname`
echo Directory is `pwd`
echo Time is `date`

#module load fenics

cd $OUTPUTDIR
cp -r $SCRIPTDIR/* . # copy everything from from SCRIPTDIR to OUTPUTDIR

# Execute computation
if [ "$JOBNAME" == "all_benchmarks" ]; then
    BENCH="."
else
    BENCH="${JOBNAME}.py"
fi
PYTHONHASHSEED=0 DOLFIN_NOPLOT=1 mpirun --display-map python -m pytest -svl $BENCH

# TODO: Make plots

# Move output file to OUTPUTDIR
cd $SCRIPTDIR
mv -f slurm-${SLURM_JOB_ID}.out $OUTPUTDIR/.

echo Time is `date`
