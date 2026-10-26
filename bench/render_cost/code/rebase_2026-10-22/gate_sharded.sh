#!/usr/bin/env bash
# Gate on tracked files only (untracked user files present). Shards pytest by file (no xdist in Isaac python).
cd /home/jumasaet/isaac-thermal-camera-simulation
PY=/home/jumasaet/isaac_sim/python.sh
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=$1; N=${2:-12}
: > $L/status
files=$(git ls-files 'src/*.py' 'tests/*.py' 'scripts/*.py')
$PY -m ruff check $files > $L/ruff.txt 2>&1; echo "ruff=$?" >> $L/status
$PY -m ruff format --check $files > $L/fmt.txt 2>&1; echo "fmt=$?" >> $L/status
$PY -m mypy src/irsim src/irsim_isaac src/irsim_eval src/irsim_viewer > $L/mypy.txt 2>&1; echo "mypy=$?" >> $L/status
$PY scripts/next_step.py --check > $L/next.txt 2>&1; echo "next=$?" >> $L/status
git ls-files 'tests/unit/*.py' 'tests/golden/*.py' | grep -E '/test_[^/]*\.py$' > $L/testfiles.txt
pids=()
for i in $(seq 0 $((N-1))); do
  awk -v n=$N -v i=$i 'NR%n==i' $L/testfiles.txt > $L/shard$i.txt
  $PY -m pytest $(cat $L/shard$i.txt) -q -m "not isaac and not gpu" -p no:cacheprovider -rfE > $L/pytest$i.txt 2>&1 &
  pids+=($!)
done
rc=0; for i in "${!pids[@]}"; do wait ${pids[$i]}; r=$?; echo "pytest$i=$r" >> $L/status; [ $r -ne 0 ] && rc=1; done
echo "pytest=$rc" >> $L/status
echo done >> $L/status
