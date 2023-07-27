Download the AKU-net.hdf5 file
Compile the model using: model.compile(optimizer= Adam(lr = 0.001, decay=1e-6), loss='binary_crossentropy', metrics=[dice_coef])
run the testing images
