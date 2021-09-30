from gym.envs.registration import register

register(
    id='oaas-v0',
    entry_point='gym_oaas.envs:oaasEnv',
)