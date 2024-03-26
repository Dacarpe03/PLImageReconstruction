import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class SimpleFullyConnectedNN(nn.Module):

	def __init__(self, input_size, output_shape, hidden_layers_sizes):
		super(SimpleFullyConnectedNN, self).__init__()

		self.output_shape = output_shape
		output_size = 2*128*128

		layers = []
		layers.append(nn.Linear(input_size, hidden_layers_sizes[0]))

		for i in range(len(hidden_layers_sizes)-1):
			layers.append(nn.Linear(hidden_layers_sizes[i], hidden_layers_sizes[i+1]))
			layers.append(nn.ReLU())

		layers.append(nn.Linear(hidden_layers_sizes[-1], output_size))
		self.model = nn.Sequential(*layers)


	def forward(self, x):

		return self.model(x)


def train_epoch(model, 
				optimizer, 
				loss_function,
				training_dataloader,
				epoch_index,
				tb_writer):
	
	running_loss = 0.
	last_loss = 0.
	
	for step, data in enumerate(training_dataloader):

		fluxes, complex_fields = data

		optimizer.zero_grad()

		outputs = model(fluxes)

		loss = loss_function(outputs, complex_fields)
		loss.backward()

		optimizer.step()

		running_loss += loss.item()

		# Report each 100 batches
		print(' batch {} loss:{}'.format(step+1, last_loss))
		if step%10 == 9:
			last_loss = running_loss/100
			print(' batch {} loss:{}'.format(step+1, last_loss))
			tb_x = epoch_index * len(training_loader)+step+1
			tb_writer.add_scalar('Loss/train', last_loss, tb_x)
			running_loss = 0.

		return last_loss


def train_model(model,
				model_name,
				epochs,
				optimizer,
				loss_function,
				training_dataloader,
				validation_dataloader
				):

	timestamp = "today"
	writer = SummaryWriter('runs/{}_{}'.format(model_name, timestamp))

	current_epoch = 0

	best_validation_loss = 1_000_000.

	for epoch in range(epochs):
		print(f"Epoch {epoch}")

		model.train(True)
		average_loss = train_epoch(model,
								   optimizer,
								   loss_function,
								   training_dataloader,
								   epoch,
								   writer)

		running_validation_loss = 0.0

		model.eval()

		with torch.no_grad():
			for i, validation_data in enumerate(validation_dataloader):
				validation_fluxes, validation_complex_fields = validation_data
				validation_outputs = model(validation_fluxes)
				validation_loss = loss_function(validation_outputs, validation_complex_fields)
				running_validation_loss += validation_loss

		average_validation_loss = running_validation_loss / (i+1)
		print('LOSS train {} validation {}'.format(average_loss, average_validation_loss))

		writer.add_scalars('Traning vs. Validation Loss',
							{'Training': average_loss, 'Validation': average_validation_loss}, 
							current_epoch)
		writer.flush()

		if average_validation_loss < best_validation_loss:
			best_validation_loss = average_validation_loss
			model_path = '{}_{}_{}'.format(model_name, timestamp, current_epoch)
			torch.save(model.state_dict(), model_path)

		current_epoch += 1