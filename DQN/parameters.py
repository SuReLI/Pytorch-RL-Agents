
class Parameters:

	###########################################################################
	# Environment settings

	DISPLAY = True
	GUI = True
	LOAD = False


	###########################################################################
	# Algorithm hyper-parameters

	GAMMA = 0.99 				# Discount

	N_STEP = 1
	GAMMA_N = GAMMA ** N_STEP

	GRAD_CLAMPING = False

	MAX_EPISODES = 1000

	###########################################################################
	# Display frequencies

	EP_REWARD_FREQ = 100
	PLOT_FREQ      = 200
	RENDER_FREQ    = 500
	SAVE_FREQ      = 0

	RESULTS_PATH = 'results/'
