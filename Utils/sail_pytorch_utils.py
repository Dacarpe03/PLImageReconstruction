import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from data_utils import normalize_data
from constants import SUBFILE_SAMPLES, TRAIN_FILE_SUFFIXES, TRAIN_CNN_AMP_PHASE_PATH, NUMPY_SUFFIX, TRAIN_COMPLEX_FIELDS_PREFIX


class DynamicCustomAmpPhaseDataset(Dataset):

	def __init__(self, file_paths, do_shuffle):
		self.item_counter = 0
		self.next_file = 0
		self.file_paths = file_paths
		self.do_shuffle = do_shuffle

		self.data = self._load_data()

	def _load_data(self):
		data = self.load_next_file()

		return data

	def __len__(self):
		return SUBFILE_SAMPLES*len(self.file_paths)

	def __getitem__(self, idx):
		idx = idx%SUBFILE_SAMPLES
		sample = self.data[idx]
		self.item_counter += 1

		# Convert numpy array to torch tensor if needed
		sample = torch.from_numpy(sample)

		if self.item_counter > SUBFILE_SAMPLES-1:
			self.load_next_file()
		return sample

	def load_next_file(self):
		self.item_counter = 0
		data = []
		if self.next_file < len(self.file_paths):
			arrays = np.load(self.file_paths[self.next_file], allow_pickle=True)

			if self.do_shuffle:
				shuffled_indices = np.random.permutation(len(arrays))
				arrays = arrays[shuffled_indices]
                
			data.extend(arrays)
			self.next_file += 1

		return data


class DynamicCustomPSFDataset(Dataset):

	def __init__(self, file_paths, do_shuffle):
		self.item_counter = 0
		self.next_file = 0
		self.file_paths = file_paths
		self.do_shuffle = do_shuffle

		self.data = self._load_data()

	def _load_data(self):
		data = self.load_next_file()

		return data

	def __len__(self):
		return SUBFILE_SAMPLES*len(self.file_paths)

	def __getitem__(self, idx):
		idx = idx%SUBFILE_SAMPLES
		sample = self.data[idx]
		self.item_counter += 1

		# Convert numpy array to torch tensor if needed
		sample = torch.from_numpy(sample)

		if self.item_counter > SUBFILE_SAMPLES-1:
			self.load_next_file()
		return sample

	def load_next_file(self):
		self.item_counter = 0
		data = []
		if self.next_file < len(self.file_paths):
			complex_arrays = np.load(self.file_paths[self.next_file], allow_pickle=True)

			real_part = np.real(complex_arrays).astype(np.float32)
			norm_real_part, scaler = normalize_data(real_part, (0, 256))
			imaginary_part = np.imag(complex_arrays).astype(np.float32)
			norm_imaginary_part, scaler = normalize_data(imaginary_part, (0, 256))

			arrays = np.stack((norm_real_part, norm_imaginary_part), axis=1)
			if self.do_shuffle:
				shuffled_indices = np.random.permutation(len(arrays))
				arrays = arrays[shuffled_indices]
                
			data.extend(arrays)
			self.next_file += 1

		return data


def instantiate_diffusion_amp_phase_dataloader(
 	batch_size
 	):
 	file_paths = []
 	for file_number in TRAIN_FILE_SUFFIXES:
 		subfile_path = f"{TRAIN_CNN_AMP_PHASE_PATH}{file_number}{NUMPY_SUFFIX}"
 		file_paths.append(subfile_path)

 	amp_phase_dynamic_dataset = DynamicCustomAmpPhaseDataset(file_paths, do_shuffle=True)
 	dynamic_custom_dataloader = DataLoader(amp_phase_dynamic_dataset, batch_size=batch_size)

 	return dynamic_custom_dataloader


def instantiate_diffusion_psf_dataloader(
 	batch_size,
 	validation=False
 	):

	if validation:
		file_paths = []
 	file_paths = []
 	for file_number in TRAIN_FILE_SUFFIXES:
 		subfile_path = f"{TRAIN_COMPLEX_FIELDS_PREFIX}{file_number}{NUMPY_SUFFIX}"
 		file_paths.append(subfile_path)

 	psf_dynamic_dataset = DynamicCustomPSFDataset(file_paths, do_shuffle=True)
 	dynamic_custom_dataloader = DataLoader(psf_dynamic_dataset, batch_size=batch_size)

 	return dynamic_custom_dataloader