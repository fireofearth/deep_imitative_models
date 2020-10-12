# export CUDA_VISIBLE_DEVICES=0; SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 python -m pdb -c c $DIMROOT/carla_agent.py \
python -m pdb -c c $DIMROOT/carla_agent.py \
    main.log_level=DEBUG \
    main.pilot=dim \
    dim.goal_likelihood=RegionIndicator \
    waypointer.interpolate=False \
    experiment.scene=Town02 \
    #experiment.scene=Town01 \
    #experiment.n_vehicles=50

    # experiment.n_vehicles=10
