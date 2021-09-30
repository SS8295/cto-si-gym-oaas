from gym.envs.registration import register

register(
    id='oaas-v0',
    entry_point='gym_oaas.envs:OaasEnv',
)

register(
    id='oaas-v2',
    entry_point='gym_oaas.envs:OaasEnv2',
)

register(
    id='oaas-multi-v0',
    entry_point='gym_oaas.envs:OaasEnvMulti',
)