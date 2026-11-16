#!/bin/bash
# deploy_aws_efa.sh — End-to-End Repeatable AWS EFA RDMA Cluster Provisioner
set -e

REGION="us-east-1"
INSTANCE_TYPE="c7gn.16xlarge"
KEY_NAME="llama-shmem-key"
KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"
CLUSTER_NAME="llama-shmem-efa-cluster"
SG_NAME="llama-shmem-efa-sg"
LOCAL_WORKSPACE="$HOME/llama-shmem-workspace"
MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
MODEL_FILE="qwen2.5-7b-instruct-q4_k_m.gguf"

SSH_OPTS="-i $KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

echo "1. Resolving latest Ubuntu 24.04 ARM64 AMI..."
AMI_ID=$(aws ssm get-parameters --region $REGION \
    --names /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id \
    --query 'Parameters[0].Value' --output text)

echo "2. Configuring EFA Placement Group & Security Group..."
aws ec2 create-placement-group --region $REGION --group-name $CLUSTER_NAME --strategy cluster 2>/dev/null || true
VPC_ID=$(aws ec2 describe-vpcs --region $REGION --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text)
SUBNET_ID=$(aws ec2 describe-subnets --region $REGION --filters "Name=default-for-az,Values=true" --query 'Subnets[0].SubnetId' --output text)

SG_ID=$(aws ec2 describe-security-group-ids --region $REGION --group-names $SG_NAME 2>/dev/null || \
        aws ec2 describe-security-groups --region $REGION --group-names $SG_NAME --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)

if [ -z "$SG_ID" ] || [ "$SG_ID" == "None" ]; then
    SG_ID=$(aws ec2 create-security-group --region $REGION --group-name $SG_NAME --description "EFA llama-shmem" --vpc-id $VPC_ID --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID --protocol tcp --port 22 --cidr 0.0.0.0/0 >/dev/null
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID --protocol all --source-group $SG_ID >/dev/null
fi

echo "3. Building EFA cloud-init payload (with OpenMPI + llama.cpp + EFA stack)..."
cat << 'EOF' > user_data.yaml
#cloud-config
package_update: true
packages:
  - build-essential
  - cmake
  - python3-pip
  - git
  - wget
  - openmpi-bin
  - libopenmpi-dev

runcmd:
  # Lock down SSH permissions immediately
  - echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config
  - chown -R ubuntu:ubuntu /home/ubuntu
  - chmod 700 /home/ubuntu/.ssh
  - chmod 600 /home/ubuntu/.ssh/authorized_keys

  # Install AWS EFA Libfabric Stack
  - cd /opt
  - curl -sO https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
  - tar -xf aws-efa-installer-latest.tar.gz
  - cd aws-efa-installer
  - ./efa_installer.sh -y -d --skip-limit-conf

  # Build and install llama.cpp shared libraries natively for Graviton3 ARM64
  - cd /opt
  - git clone https://github.com/ggerganov/llama.cpp.git
  - cd llama.cpp
  - cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON
  - cmake --build build --config Release -j $(nproc)
  - cmake --install build
  - ldconfig

  # Write completion sentinel
  - touch /tmp/cloud_init_done
EOF

echo "4. Launching 2x $INSTANCE_TYPE Instances (128 vCPUs total, 30GB GP3 NVMe)..."
INSTANCE_IDS=$(aws ec2 run-instances \
    --region $REGION --image-id $AMI_ID --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --placement "GroupName=$CLUSTER_NAME" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
    --network-interfaces "[{\"DeviceIndex\":0,\"InterfaceType\":\"efa\",\"Groups\":[\"$SG_ID\"],\"SubnetId\":\"$SUBNET_ID\"}]" \
    --count 2 --user-data file://user_data.yaml \
    --query 'Instances[*].InstanceId' --output text)

rm -f user_data.yaml
echo "   Instances launched: $INSTANCE_IDS"
echo "   Waiting for EC2 running state..."
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_IDS

IPS=$(aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_IDS \
    --query 'Reservations[*].Instances[*].[PublicIpAddress,PrivateIpAddress]' --output text)

NODE0_PUB=$(echo "$IPS" | head -n 1 | awk '{print $1}')
NODE0_PRIV=$(echo "$IPS" | head -n 1 | awk '{print $2}')
NODE1_PUB=$(echo "$IPS" | tail -n 1 | awk '{print $1}')
NODE1_PRIV=$(echo "$IPS" | tail -n 1 | awk '{print $2}')

echo "   Node 0 (Master): Public=$NODE0_PUB | Private=$NODE0_PRIV"
echo "   Node 1 (Worker): Public=$NODE1_PUB | Private=$NODE1_PRIV"

echo "5. Waiting for SSH & cloud-init compilation (EFA + llama.cpp) to finish on both nodes..."
for HOST_IP in "$NODE0_PUB" "$NODE1_PUB"; do
    echo -n "   Polling $HOST_IP "
    until ssh $SSH_OPTS -o ConnectTimeout=5 ubuntu@$HOST_IP "test -f /tmp/cloud_init_done" 2>/dev/null; do
        echo -n "."
        sleep 10
    done
    echo " Ready!"
done

echo "6. Syncing clean source code to Master (excluding local x86 build/ and *.gguf)..."
rsync -avz --delete \
    --exclude='build/' \
    --exclude='*.gguf' \
    --exclude='.git/' \
    -e "ssh $SSH_OPTS" \
    "$LOCAL_WORKSPACE/" ubuntu@$NODE0_PUB:~/llama-shmem-workspace/

scp $SSH_OPTS "$KEY_PATH" ubuntu@$NODE0_PUB:~/.ssh/llama-shmem-key.pem

echo "7. Building native ARM64 engine on Master, pulling 7B GGUF, and syncing to Worker..."
ssh $SSH_OPTS ubuntu@$NODE0_PUB << REMOTE_EOF
set -e

# Configure passwordless intra-cluster SSH
chmod 400 ~/.ssh/llama-shmem-key.pem
cat << 'CFG' > ~/.ssh/config
Host 172.31.*
    IdentityFile ~/.ssh/llama-shmem-key.pem
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
CFG
chmod 600 ~/.ssh/config

# Clean native ARM64 build
mkdir -p ~/llama-shmem-workspace/build
cd ~/llama-shmem-workspace/build
rm -rf *
cmake ..
make -j\$(nproc)

echo "Verifying binary architecture:"
file ~/llama-shmem-workspace/build/llama_shmem

# Download 7B GGUF weights if not present
cd ~/llama-shmem-workspace
if [ ! -f "$MODEL_FILE" ]; then
    wget -q --show-progress "$MODEL_URL" -O "$MODEL_FILE"
fi

# Sync compiled ARM64 workspace + model weights to Worker over internal network
rsync -avz --progress ~/llama-shmem-workspace/ ubuntu@$NODE1_PRIV:~/llama-shmem-workspace/
REMOTE_EOF

echo -e "\n================================================================================"
echo "EFA RDMA Cluster Provisioned, Compiled (ARM64), and Synced!"
echo "Master Public IP:  $NODE0_PUB"
echo "Master Private IP: $NODE0_PRIV"
echo "Worker Private IP: $NODE1_PRIV"
echo "--------------------------------------------------------------------------------"
echo "1. SSH into Master:"
echo "   ssh -i ~/.ssh/llama-shmem-key.pem ubuntu@$NODE0_PUB"
echo ""
echo "2. Run the 16-thread Tensor Parallel RDMA benchmark:"
echo "   oshrun -np 2 --host $NODE0_PRIV:1,$NODE1_PRIV:1 \\"
echo "     -x FI_PROVIDER=efa \\"
echo "     -x FI_EFA_USE_DEVICE_RDMA=1 \\"
echo "     -x LD_LIBRARY_PATH=/usr/local/lib:/opt/amazon/efa/lib \\"
echo "     setarch aarch64 -R ~/llama-shmem-workspace/build/llama_shmem \\"
echo "     ~/llama-shmem-workspace/$MODEL_FILE \\"
echo "     \"Explain the concept of speculative decoding in distributed LLM inference.\" 128 0.0 0.0 42 16"
echo "================================================================================"
