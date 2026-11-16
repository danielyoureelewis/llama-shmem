#!/bin/bash
# auto_deploy_llama_shmem.sh

REGION="us-east-1"
# Dynamically fetch the latest official Ubuntu 24.04 LTS ARM64 AMI
AMI_ID=$(aws ssm get-parameters --region $REGION --names /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id --query 'Parameters[0].Value' --output text)
INSTANCE_TYPE="c7g.large"
KEY_NAME="llama-shmem-key"
CLUSTER_NAME="llama-shmem-tcp-cluster"
SG_NAME="llama-shmem-sg"

echo "1. Generating ephemeral intra-cluster ed25519 keys..."
rm -f cluster_key cluster_key.pub
ssh-keygen -t ed25519 -f cluster_key -N "" -q
PRIV_KEY=$(cat cluster_key | sed 's/^/      /')
PUB_KEY=$(cat cluster_key.pub)

echo "2. Configuring Placement Group & Security Group..."
aws ec2 create-placement-group --region $REGION --group-name $CLUSTER_NAME --strategy cluster 2>/dev/null
VPC_ID=$(aws ec2 describe-vpcs --region $REGION --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text)
SG_ID=$(aws ec2 create-security-group --region $REGION --group-name $SG_NAME --description "llama-shmem" --vpc-id $VPC_ID --query 'GroupId' --output text 2>/dev/null)

# Fixed the typo here
if [ -z "$SG_ID" ]; then
    SG_ID=$(aws ec2 describe-security-groups --region $REGION --group-names $SG_NAME --query "SecurityGroups[0].GroupId" --output text)
else
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID --protocol tcp --port 22 --cidr 0.0.0.0/0 2>/dev/null
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG_ID --protocol all --source-group $SG_ID 2>/dev/null
fi

echo "3. Building cloud-init payload..."
cat << EOF > user_data.yaml
#cloud-config
package_update: true
packages:
  - build-essential
  - cmake
  - openmpi-bin
  - libopenmpi-dev
  - python3-pip

write_files:
  - path: /home/ubuntu/.ssh/id_ed25519
    owner: ubuntu:ubuntu
    permissions: '0600'
    content: |
${PRIV_KEY}
  - path: /home/ubuntu/.ssh/config
    owner: ubuntu:ubuntu
    permissions: '0600'
    content: |
      Host *
        StrictHostKeyChecking no
        UserKnownHostsFile /dev/null

runcmd:
  - echo "${PUB_KEY}" >> /home/ubuntu/.ssh/authorized_keys
  - echo "* soft memlock unlimited" >> /etc/security/limits.conf
  - echo "* hard memlock unlimited" >> /etc/security/limits.conf
  - echo 'memheap_base_max_segments = 128' >> /etc/openmpi/openmpi-mca-params.conf
  - pip3 install huggingface_hub --break-system-packages
EOF

echo "4. Launching 2 Instances with AMI $AMI_ID..."
INSTANCE_IDS=$(aws ec2 run-instances \
    --region $REGION --image-id $AMI_ID --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME --security-group-ids $SG_ID \
    --placement "GroupName=$CLUSTER_NAME" \
    --instance-market-options '{"MarketType":"spot"}' \
    --count 2 --user-data file://user_data.yaml \
    --query 'Instances[*].InstanceId' --output text)

echo "Waiting for instances to enter 'running' state ($INSTANCE_IDS)..."
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_IDS

IPS=$(aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_IDS \
    --query 'Reservations[*].Instances[*].[PublicIpAddress,PrivateIpAddress]' --output text)

NODE0_PUB=$(echo "$IPS" | head -n 1 | awk '{print $1}')
NODE0_PRIV=$(echo "$IPS" | head -n 1 | awk '{print $2}')
NODE1_PRIV=$(echo "$IPS" | tail -n 1 | awk '{print $2}')

rm cluster_key cluster_key.pub user_data.yaml

echo -e "\n============================================="
echo "Deployment Complete!"
echo "Node 0 (Master) Public IP:  $NODE0_PUB"
echo "Node 0 (Master) Private IP: $NODE0_PRIV"
echo "Node 1 (Worker) Private IP: $NODE1_PRIV"
echo "============================================="
echo -e "\nRun this on ai-server to push your code and build on AWS Node 0:"
echo "rsync -avz -e 'ssh -i ~/.ssh/$KEY_NAME.pem -o StrictHostKeyChecking=no' ~/llama-shmem-workspace/ ubuntu@$NODE0_PUB:~/llama-shmem-workspace/"
echo "ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@$NODE0_PUB 'cd ~/llama-shmem-workspace/build && cmake .. && make -j\$(nproc)'"
echo -e "\nThen download the model and launch the distributed run from AWS Node 0:"
echo "ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@$NODE0_PUB"
echo "huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir ~/llama-shmem-workspace"
echo "oshrun -np 2 --host $NODE0_PRIV:1,$NODE1_PRIV:1 -x UCX_TLS=tcp,self setarch aarch64 -R ~/llama-shmem-workspace/build/llama_shmem ~/llama-shmem-workspace/qwen2.5-0.5b-instruct-q4_k_m.gguf \"The capital of France is\" 64 0.8 0.05 42"
