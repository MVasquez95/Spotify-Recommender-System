from lightfm import LightFM

def train_lightfm(interactions, loss="warp", epochs=10, num_threads=4):
    model = LightFM(loss=loss)
    model.fit(interactions, epochs=epochs, num_threads=num_threads)
    return model