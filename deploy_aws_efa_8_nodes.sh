#!/bin/bash
# deploy_aws_efa_8_nodes.sh — 8-Node Physical Cluster (8x c7gn.4xlarge = 128 vCPUs) for Qwen2.5-32B
set -e

REGION="us-east-1"
NUM_NODES=8
INSTANCE_TYPE="c7gn.4xlarge"   # 16 ARM64 cores, 32GB DDR5, 50Gbps ENA (128 vCPUs total = exact quota match)
KEY_NAME="llama-shmem-key"
KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"
CLUSTER_NAME="llama-shmem-efa-cluster"
SG_NAME="llama-shmem-efa-sg"
LOCAL_WORKSPACE="$HOME/llama-shmem-workspace"

MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
MODEL_FILE="qwen2.5-32b-instruct-q4_k_m.gguf"
DRAFT_URL="https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
DRAFT_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"

SSH_OPTS="-i $KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

echo "1. Resolving latest Ubuntu 24.04 ARM64 AMI..."
AMI_ID=$(aws ssm get-parameters --region $REGION \
    --names /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id \
    --query 'Parameters[0].Value' --output text)

echo "2. Configuring Cluster Placement Group & Security Group..."
aws ec2 create-placement-group --region $REGION --group-name $CLUSTER_NAME --strategy cluster 2>/dev/null || true
VPC_ID=$(aws ec2 describe-vpcs --region $REGION --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text)
SUBNET_ID=$(aws ec2 describe-subnets --region $REGION --filters "Name=default-for-az,Values=true" --query 'Subnets[0].SubnetId' --output text)

SG_ID=$(aws ec2 describe-security-groups --region $REGION --group-names $SG_NAME --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)
if [ -z "$SG_ID" ] || [ "$SG_ID" == "None" ]; then
    SG_ID=$(aws ec2 create-security-group --region $REGION --group-name $SG_NAME --description "llama-shmem cluster" --vpc-id $VPC_ID --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID --protocol tcp --port 22 --cidr 0.0.0.0/0 >/dev/null
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID --protocol all --source-group $SG_ID >/dev/null
fi

echo "3. Building 8-Node cloud-init payload (Parallel 32B GGUF Download + OpenMPI + llama.cpp)..."
cat << CLOUD_EOF > user_data.yaml
#cloud-config
package_update: true
packages:
  - build-essential
  - cmake
  - git
  - wget
  - aria2
  - openmpi-bin
  - libopenmpi-dev
  - libfabric-dev

runcmd:
  - echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config
  - mkdir -p /home/ubuntu/llama-shmem-workspace
  - chown -R ubuntu:ubuntu /home/ubuntu
  - chmod 700 /home/ubuntu/.ssh
  - chmod 600 /home/ubuntu/.ssh/authorized_keys

  # Start parallel multi-connection download of 19.8GB 32B GGUF + 0.5B Draft GGUF in background
  - su - ubuntu -c "aria2c -x 8 -s 8 -d /home/ubuntu/llama-shmem-workspace -o $MODEL_FILE '$MODEL_URL' &"
  - su - ubuntu -c "aria2c -x 4 -s 4 -d /home/ubuntu/llama-shmem-workspace -o $DRAFT_FILE '$DRAFT_URL' &"

  # Build and install llama.cpp shared libraries natively for Graviton3 ARM64
  - cd /opt
  - git clone https://github.com/ggerganov/llama.cpp.git
  - cd llama.cpp
  - cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON
  - cmake --build build --config Release -j \$(nproc)
  - cmake --install build
  - ldconfig

  # Wait for background aria2c model downloads to finish
  - while pgrep -x aria2c >/dev/null; do sleep 5; done
  - chown -R ubuntu:ubuntu /home/ubuntu/llama-shmem-workspace
  - touch /tmp/cloud_init_done
CLOUD_EOF

echo "4. Launching ${NUM_NODES}x $INSTANCE_TYPE Instances (128 vCPUs total, 50GB GP3 NVMe per node)..."
INSTANCE_IDS=$(aws ec2 run-instances \
    --region $REGION --image-id $AMI_ID --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --placement "GroupName=$CLUSTER_NAME" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
    --network-interfaces "[{\"DeviceIndex\":0,\"InterfaceType\":\"interface\",\"Groups\":[\"$SG_ID\"],\"SubnetId\":\"$SUBNET_ID\"}]" \
    --count $NUM_NODES --user-data file://user_data.yaml \
    --query 'Instances[*].InstanceId' --output text)

rm -f user_data.yaml
echo "   Instances launched: $INSTANCE_IDS"
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_IDS

PUB_IPS=($(aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_IDS --query 'Reservations[*].Instances[*].PublicIpAddress' --output text))
PRIV_IPS=($(aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_IDS --query 'Reservations[*].Instances[*].PrivateIpAddress' --output text))

MASTER_PUB=${PUB_IPS[0]}
MASTER_PRIV=${PRIV_IPS[0]}

HOST_SLOTS=""
for ip in "${PRIV_IPS[@]}"; do
    HOST_SLOTS="${HOST_SLOTS:+$HOST_SLOTS,}$ip:1"
done

echo "   Master Public IP: $MASTER_PUB | Master Private IP: $MASTER_PRIV"
echo "   8-Node Hostfile:  $HOST_SLOTS"

echo "5. Waiting for llama.cpp build + 32B GGUF download on all $NUM_NODES nodes..."
for HOST_IP in "${PUB_IPS[@]}"; do
    echo -n "   Polling $HOST_IP "
    until ssh $SSH_OPTS -o ConnectTimeout=5 ubuntu@$HOST_IP "test -f /tmp/cloud_init_done" 2>/dev/null; do
        echo -n "."
        sleep 10
    done
    echo " Ready!"
done

echo "6. Syncing clean C++ source code to Master..."
rsync -avz \
    --exclude='build/' \
    --exclude='*.gguf' \
    --exclude='.git/' \
    -e "ssh $SSH_OPTS" \
    "$LOCAL_WORKSPACE/" ubuntu@$MASTER_PUB:~/llama-shmem-workspace/

scp $SSH_OPTS "$KEY_PATH" ubuntu@$MASTER_PUB:~/.ssh/llama-shmem-key.pem

echo "7. Compiling ARM64 binary on Master and distributing to all 7 Workers..."
WORKER_PRIVS="${PRIV_IPS[@]:1}"
ssh $SSH_OPTS ubuntu@$MASTER_PUB << REMOTE_EOF
set -e
chmod 400 ~/.ssh/llama-shmem-key.pem
cat << 'CFG' > ~/.ssh/config
Host 172.31.*
    IdentityFile ~/.ssh/llama-shmem-key.pem
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
CFG
chmod 600 ~/.ssh/config

mkdir -p ~/llama-shmem-workspace/build
cd ~/llama-shmem-workspace/build
rm -rf *
cmake ..
make -j\$(nproc)

for wip in $WORKER_PRIVS; do
    ssh ubuntu@\$wip "mkdir -p ~/llama-shmem-workspace/build"
    rsync -az ~/llama-shmem-workspace/build/llama_shmem ubuntu@\$wip:~/llama-shmem-workspace/build/ &
done
wait
REMOTE_EOF

echo -e "\n================================================================================"
echo "8-Node Cluster Ready with Qwen2.5-32B-Instruct-Q4_K_M!"
echo "Master Public IP: $MASTER_PUB"
echo "--------------------------------------------------------------------------------"
echo "1. SSH into Master:"
echo "   ssh -i ~/.ssh/llama-shmem-key.pem ubuntu@$MASTER_PUB"
echo ""
echo "2. Run 8-Node Pure Tensor Parallelism (k_draft=0, B_VERIFY=1):"
echo "   oshrun -np 8 --host $HOST_SLOTS \\"
echo "     --mca btl tcp,self \\"
echo "     -x LD_LIBRARY_PATH=/usr/local/lib \\"
echo "     setarch aarch64 -R ~/llama-shmem-workspace/build/llama_shmem \\"
echo "     ~/llama-shmem-workspace/$MODEL_FILE \\"
echo "     \"Explain the concept of speculative decoding in distributed LLM inference.\" 128 0.0 0.0 42 4 0"
echo ""
echo "3. Run 8-Node TP + 0.5B Neural Speculative Decoding (k_draft=1, B_VERIFY=2):"
echo "   oshrun -np 8 --host $HOST_SLOTS \\"
echo "     --mca btl tcp,self \\"
echo "     -x LD_LIBRARY_PATH=/usr/local/lib \\"
echo "     setarch aarch64 -R ~/llama-shmem-workspace/build/llama_shmem \\"
echo "     ~/llama-shmem-workspace/$MODEL_FILE \\"
echo "     \"Explain the concept of speculative decoding in distributed LLM inference.\" 128 0.0 0.0 42 4 1 \\"
echo "     ~/llama-shmem-workspace/$DRAFT_FILE"
echo "================================================================================"
