import numpy as np

names = [
"ld_20M_zernike_complex_fields.npy",
"ld_14M_zernike_complex_fields.npy",
"ld_9M_zernike_complex_fields.npy",
"ld_5M_zernike_complex_fields.npy",
"ld_2M_zernike_complex_fields.npy",
"ld_2M_zernike_intensities.npy",
"ld_5M_zernike_intensities.npy",
"ld_9M_zernike_intensities.npy",
"ld_14M_zernike_intensities.npy",
"ld_20M_zernike_intensities.npy",
]

for name in names:
	vector = np.load(name)

	start = 0
	end = 10000
	jump = 10000
	for i in range(7):
		new_name = name.replace(".npy", f"_{i}.npy")
		new_vector = vector[start:end]
		print(new_vector.shape)
		np.save(new_name, new_vector)
		start += jump
		end += jump
