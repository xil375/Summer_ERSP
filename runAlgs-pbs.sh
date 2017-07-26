#Sample PBS job script
#
# Copy this script, customize it and then submit it with the ``qsub''
# command. For example:
#
# cp pbs-template.sh myjob-pbs.sh
# {emacs|vi} myjob-pbs.sh
# qsub myjob-pbs.sh
#
# PBS directives are fully documented in the ``qsub'' man page. Directives
# may be specified on the ``qsub'' command line, or embedded in the
# job script.
#
# For example, if you want the batch job to inherit all your environment
# variables, use the ``V'' switch when you submit the job script:
#
# qsub -V myjob-pbs.sh
#
# or uncomment the following line by removing the initial ``###''
#PBS -V

# Note: group all PBS directives at the beginning of your script.
# Any directives placed after the first shell command will be ignored.

### Set the job name
#PBS -N runAlgs

### Run in the queue named "batch"
###PBS -q route
### Use the bourne shell
#PBS -S /bin/sh

### Remove only the three initial "#" characters before #PBS
### in the following lines to enable:
###
### To send email when the job is completed:
#PBS -m ae
#PBS -M xil375@ucsd.edu

### Optionally set the destination for your program's output
### Specify localhost and an NFS filesystem to prevent file copy errors.
#PBS -e localhost:$HOME/Result/myjob.err_yat2
#PBS -o localhost:$HOME/Result/myjob.log_yat2
## $PBS_JOBID
### Specify the number of cpus for your job.  This example will allocate 4 cores
### using 2 processors on each of 2 nodes.
#PBS -l nodes=1:ppn=1

### Tell PBS how much memory you expect to use. Use units of 'b','kb', 'mb' or 'gb'.
#PBS -l mem=3gb

### Tell PBS the anticipated run-time for your job, where walltime=HH:MM:SS
#PBS -l walltime=24:00:00

### Switch to the working directory; by default TORQUE launches processes
### from your home directory.
cd $PBS_O_WORKDIR
echo Working directory is $PBS_O_WORKDIR

# Calculate the number of processors allocated to this run.
NPROCS=`wc -l < $PBS_NODEFILE`

# Calculate the number of nodes allocated.
NNODES=`uniq $PBS_NODEFILE | wc -l`

### Display the job context
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Using ${NPROCS} processors across ${NNODES} nodes

### OpenMPI will automatically launch processes on all allocated nodes.
## MPIRUN=`which mpirun`
## ${MPIRUN} my-openmpi-program

### Or, just run your serial program
#module load python_2.7.3
python MultiClass_MainScript.py
