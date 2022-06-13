import json
import pickle
import numpy as np

__data_columns = None
__locations = None
__model = None


def get_estimated_price(loc, sqft, BHK, bath) -> float:
    global __data_columns, __locations, __model
    try:
        loc_index = __data_columns.index(loc.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0], 2)


def get_location_names():
    return __locations


def load_saved_artifacts():
    global __data_columns
    global __locations
    global __model

    with open("./server/artifacts/columns.json", "r") as f:
        jsonf = json.load(f)
        __data_columns = jsonf['data_columns']
        __locations = __data_columns[3:]

    with open("./server/artifacts/banglore_houses_model.pickle", "rb") as f:
        __model = pickle.load(f)


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price("1st Phase JP Nagar", 1000, 3, 3))
