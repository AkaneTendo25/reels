Base.@kwdef struct TrainingJob{B<:AbstractVideoModelBackend,M,F}
    backend::B
    model::M
    batch::F
    training::TrainingConfig
    output_dir::String
    checkpoint::CheckpointConfig = CheckpointConfig()
end

function train!(job::TrainingJob; resume_from=nothing, callback=nothing)
    params = [job.model.projection.A, job.model.projection.B]
    if resume_from === nothing
        rng = Xoshiro(job.training.seed)
        state = TrainingState(0, 0, rng, AdamWState(params))
    else
        restored = load_checkpoint(resume_from)
        for (p, restored_param) in zip(params, restored.params)
            copyto!(p, restored_param)
        end
        state = restored.state
    end
    opt = AdamW(learning_rate=job.training.learning_rate,
        weight_decay=job.training.weight_decay)
    accum = [zeros(Float32, size(p)) for p in params]
    losses = Float32[]
    while state.step < job.training.steps
        fill!.(accum, 0f0)
        loss = 0f0
        for _ in 1:job.training.gradient_accumulation
            batch = job.batch(state.rng)
            pred = predict_velocity(job.backend, job.model, batch.x, nothing)
            err = pred .- batch.target
            loss += sum(abs2, err) / length(err)
            dy = (2f0 / length(err)) .* err
            grad = lora_backward(job.model.projection, batch.x, dy)
            accum[1] .+= grad.A; accum[2] .+= grad.B
            state.micro_step += 1
        end
        invacc = 1f0 / job.training.gradient_accumulation
        accum .*= invacc
        norm = sqrt(sum(sum(abs2, g) for g in accum))
        if norm > job.training.max_gradient_norm
            accum .*= job.training.max_gradient_norm / norm
        end
        lr = learning_rate(job.training.learning_rate, job.training.scheduler,
            state.step + 1, job.training.steps)
        update!(opt, state.optimizer, params, accum; learning_rate=lr)
        state.step += 1
        push!(losses, loss * invacc)
        callback === nothing || callback(state, losses[end])
        if state.step % job.checkpoint.every_steps == 0
            save_checkpoint(joinpath(job.output_dir, "checkpoint-$(state.step).reels"),
                state, params)
        end
    end
    (state=state, losses=losses)
end
