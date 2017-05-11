from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD

from model import baseline_model
import settings as s


lr = 0.01


def train_baseline(X, Y):
    global lr
    model = baseline_model()
    batch_size, epochs, lr, decay = s.batch_size, s.epochs, s.lr, s.decay

    # Train model using SGD
    print batch_size, epochs, lr, decay
    sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(X, Y,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=[LearningRateScheduler(lr_schedule),
                         ModelCheckpoint(str('./trained_models/baseline_'+str(epochs)+'_'+str(batch_size)+'_'+str(lr)+'.h5'), save_best_only=True)]
              )
    return model


def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))
