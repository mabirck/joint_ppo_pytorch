# joint_ppo_pytorch
Extension of https://github.com/ikostrikov/pytorch-a2c-ppo-acktr, making it feasible to run train on multiple games simultaneously.

# Send to Docker

sudo docker build -f retro_ppo_test.docker -t $DOCKER_REGISTRY/retro_ppo_test:v1 .
sudo docker push $DOCKER_REGISTRY/retro_ppo_test:v1
sudo docker pull openai/retro-env
sudo docker pull openai/retro-agent:pytorch
sudo docker tag openai/retro-env remote-env


# PPO
python main.py --env-name "SonicTheHedgehog-Genesis,GreenHillZone.Act1" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1