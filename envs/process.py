from agents.agent import Agent

def process(thread, method):
    local_agent = Agent(
        method.main_model_factory,
        method.curiosity_model_factory,
        method.save_path,
    )

    worker = method.worker_factory.create(local_agent)

    for ciclo in range(method.episodes):
        rollout = worker.run()

        dict_metrics = method.update(rollout, local_agent)

        method.update_metrics(
            thread,
            ciclo,
            dict_metrics
        )

    method.save_models(thread)
    method.draw_plots(thread)