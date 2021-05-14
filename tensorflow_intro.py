import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

observations = 1000
xs = np.random.uniform(-10,10,(observations,1))
zs = np.random.uniform(-10,10,(observations,1))
noise = np.random.uniform(-1,1,(observations,1))
generated_inputs = np.column_stack((xs,zs))
generated_targets = 2*xs - 3*zs + 5 + noise

np.savez('TF_intro',inputs=generated_inputs, targets=generated_targets)

training_data = np.load('TF_intro.npz')
input_size = 2
output_size = 1
"""
here the weights and bias are left to tensorflow to decide but we can also use kernel & bias initializer parameters to do so.
"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        output_size,
        #kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
        #bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
        ) 
])
"""
The learning rate is left to tensorflow to decide if we don't specify
using optimizers.SGD method we have the ability to override default learning rate.
similarly loss function can also be customized.
"""
custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
model.fit(training_data['inputs'],training_data['targets'], epochs=100, verbose=True)

print('----------------------------------------------------')
weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]
print(weights)
print(bias)
print('----------------------------------------------------')

print(model.predict_on_batch(training_data['inputs']).round(1))
print(training_data['targets'])

plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
plt.xlabel('outputs')
plt.ylabel('targets')
plt.savefig('graphs/tensorflow_intro.png')