#blbalbaklab

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()