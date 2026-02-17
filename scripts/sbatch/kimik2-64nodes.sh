EXEC_DATE=$(date +%Y-%m-%d_%H-%M)
EXP=cpu-replica-rdma-direct-kimi
JOBID=6405768

srun --jobid=${JOBID} --nodes=64 --ntasks-per-node=1 --overlap bash -c "
    pid=\$(enroot list -f | grep 'pyxis_${JOBID}' -A 1 | awk '\$2 ~ /^[0-9]+\$/ {print \$2; exit}')

    if [ -z \"\$pid\" ]; then
    echo \"[ERROR] No container found on \$(hostname)\"
    exit 1
    fi

    enroot exec \"\$pid\" bash -c \"
    . /root/env.sh

    LOG_DIR='/data/logs/kimik2/${EXEC_DATE}-${EXP}'
    mkdir -p \\\$LOG_DIR
    export MILES_LOG_DIR=\\\$LOG_DIR

    cd /root/miles
    git fetch lt --quiet && git reset --hard lt/jd/rdma-cpu-replica-direct

    # Clean stale profiler traces before run
    rm -rf /root/rdma_profiler_logs

    python /root/miles/tests/test_weight_transfer_moe_multinode_kimik2_64nodes.py \\
        --multinode --mode rdma \\
        --head-node-ip \\\$HEAD_NODE_IP --nnodes \\\$NNODES --node-rank \\\$NODE_RANK \\
        --enable-nccl-nvls --released-mc-transfer-timeout --wait-after --bucket-size 1 \\
        2>&1 | tee \\\$LOG_DIR/node_\\\$NODE_RANK.log

    if [ \\\$NODE_RANK -eq 0 ]; then
        [ -d /root/rdma_profiler_logs ] && cp -r /root/rdma_profiler_logs \\\$LOG_DIR/
        if [ -f \\\$LOG_DIR/miles_timer_0.log ]; then
        python /root/miles/consolidate_timer_log.py \\\$LOG_DIR/miles_timer_0.log
        echo 'Timer log consolidated'
        fi
        echo 'Profiler extraction complete'
        ls -la \\\$LOG_DIR/
    fi
    \"
"
