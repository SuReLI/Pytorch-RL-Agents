
class Config:

    GAME = "LunarLanderContinuous-v2"
    GAME = "Pendulum-v0"

    MEMORY_CAPACITY = 1000000
    BATCH_SIZE = 8
    GAMMA = 0.99
    LEARNING_RATE_CRITIC = 0.001
    LEARNING_RATE_ACTOR = 0.001
    TAU = 0.005

    EXPLO_SIGMA = 0.1  # Exploration noise
    UPDATE_SIGMA = 0.2
    UPDATE_CLIP = 0.5

    MAX_EPISODES = 1000
    MAX_STEPS = 10000
