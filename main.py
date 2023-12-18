from model import *

# Create an instance of your model
model = DTA_model(smi_vocab, amino_vocab, 128)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Fit the model to your data

main = make_model_data('KIBA')

train_drug, train_target, train_label = main.model_data('train')

valid_drug, valid_target, valid_label = main.model_data('valid')
#
test_drug, test_target, test_label = main.model_data('test')

model.fit(x=(train_drug, train_target), y=train_label, batch_size=128,epochs=10,validation_data=[(valid_drug,valid_target),valid_label])