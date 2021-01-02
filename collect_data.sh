## collect data for dataset generation

python $DIMROOT/carla_agent.py \
    main.pilot=auto \
    data.save_data=True \
    experiment.scene="Town01" \
    experiment.n_vehicles=50 \
    experiment.seed=3267 \
    plotting.plot=False \
    experiment.frames_per_episode=210 \
    experiment.n_episodes=50 \
    experiment.min_takeover_frames=31 \

    # experiment.n_episodes=1000 \
