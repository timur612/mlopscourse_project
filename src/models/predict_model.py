model = SimpleModel.load_from_checkpoint("best_model.ckpt")
model.eval()
x = torch.randn(1, 64)

with torch.no_grad():
    y_hat = model(x)