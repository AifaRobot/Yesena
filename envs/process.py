def process(method, barrier, shared_state):
    worker = method.worker_factory.create(method.get_agent())

    for cicle in range(method.episodes):
        rollout = worker.run()

        shared_state.update_agent(rollout, method, cicle)

        barrier.wait()