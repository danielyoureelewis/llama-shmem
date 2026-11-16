#!/bin/bash
# teardown_aws_efa.sh — Terminates all running llama-shmem EFA instances
REGION="us-east-1"
INSTANCE_IDS=$(aws ec2 describe-instances --region $REGION \
    --filters "Name=key-name,Values=llama-shmem-key" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[*].Instances[*].InstanceId' --output text)

if [ -n "$INSTANCE_IDS" ]; then
    echo "Terminating instances: $INSTANCE_IDS"
    aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_IDS
else
    echo "No active llama-shmem instances found."
fi
