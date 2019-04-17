class Config:

    # GAME = "Pendulum-v0"
    GAME = "MountainCarContinuous-v0"

    MEMORY_CAPACITY = 1000000
    BATCH_SIZE = 64
    GAMMA = 0.99
    LEARNING_RATE_CRITIC = 0.001
    LEARNING_RATE_ACTOR = 0.001
    TAU = 0.001

    EPSILON = 0.001  # Exploration noise

    MAX_EPISODES = 1000
    MAX_STEPS = 200  # Max steps per episode
