# Sourced by the sbatch scripts -- not executable on its own.
#
# Restores the venv (and optionally the dataset) from single-file archives in
# $HOME onto the compute node's local /tmp. $HOME is quota'd on inodes, not bytes,
# so many-small-files things (a venv, an image dataset) cannot live there; /tmp on
# the compute node is 9.8G and unquota'd, but is wiped when the job ends.
#
# Expects REPO to be set and the caller to have already cd'd into it.

LOCAL_ROOT="/tmp/$USER"
VENV="$LOCAL_ROOT/venv"
VENV_ARCHIVE="${VENV_ARCHIVE:-$HOME/venv.tar.gz}"

mkdir -p logs "$LOCAL_ROOT"

# The marker is written only after a full extract (or a full build), so a venv
# that another job is still pip-installing into -- we may share a node with it --
# is correctly treated as absent rather than used half-built.
if [[ ! -f "$VENV/.ready" ]]; then
    if [[ ! -f "$VENV_ARCHIVE" ]]; then
        echo "ERROR: $VENV_ARCHIVE not found. Build it first:" >&2
        echo "         sbatch scripts/build_env.sbatch" >&2
        exit 1
    fi
    echo "restoring venv: $VENV_ARCHIVE -> $VENV"
    rm -rf "$VENV"
    tar xzf "$VENV_ARCHIVE" -C "$LOCAL_ROOT"
    touch "$VENV/.ready"
fi
source "$VENV/bin/activate"

# Dataset, same trick. Point DATA_ARCHIVE at a tar in $HOME whose members are
# <dataset_name>/{real,fake}/... -- e.g. built once with:
#   tar cf ~/cifake.tar -C data/raw cifake
DATA_ROOT="$REPO/data/raw"
if [[ -n "${DATA_ARCHIVE:-}" ]]; then
    echo "restoring dataset: $DATA_ARCHIVE -> $LOCAL_ROOT/raw"
    mkdir -p "$LOCAL_ROOT/raw"
    tar xf "$DATA_ARCHIVE" -C "$LOCAL_ROOT/raw"
    DATA_ROOT="$LOCAL_ROOT/raw"
fi

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "venv=$VENV data_root=$DATA_ROOT"
